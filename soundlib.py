from typing import Dict, List, NamedTuple
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time

class StreamInfo(NamedTuple):
    stream: sd.OutputStream
    timestamp: float

class IIRFilter:
    """Applies a second-order low-shelf filter to the input audio.

    This filter boosts frequencies below a cutoff frequency (around 150Hz by default).
    It uses a direct form II transposed structure for processing.

    Attributes:
        b (np.ndarray): Numerator coefficients of the filter.
        a (np.ndarray): Denominator coefficients of the filter.
        state (np.ndarray): Internal state of the filter, used for processing.

    """

    def __init__(self, samplerate: int, boost: float = 1.5):
        """Initializes the IIR filter.

        Args:
            samplerate (int): The audio sample rate.
            boost (float, optional): The gain boost in dB. Defaults to 1.5.

        """
        # Calculate coefficients for ~150Hz boost
        f = 150.0
        w0 = 2 * np.pi * (f / samplerate)
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / 2

        A = np.power(10, boost/40)

        # Filter coefficients (second-order low shelf)
        b0 = A * ((A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
        b1 = 2*A * ((A-1) - (A+1)*cos_w0)
        b2 = A * ((A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
        a0 = (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
        a1 = -2 * ((A-1) + (A+1)*cos_w0)
        a2 = (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

        # Normalize coefficients
        self.b = np.array([b0/a0, b1/a0, b2/a0])
        self.a = np.array([1.0, a1/a0, a2/a0])
        self.state = np.zeros((2, 2))  # 2 channels, 2 delay elements each

    def process(self, x):
        """Processes audio samples through the filter.

        Args:
            x (np.ndarray): Input audio data (2D array, samples x channels).

        Returns:
            np.ndarray: Filtered audio data.

        """
        y = np.zeros_like(x)
        for ch in range(x.shape[1]):
            for n in range(len(x)):
                w0 = x[n,ch] - self.a[1]*self.state[ch,0] - self.a[2]*self.state[ch,1]
                y[n,ch] = self.b[0]*w0 + self.b[1]*self.state[ch,0] + self.b[2]*self.state[ch,1]
                self.state[ch,1] = self.state[ch,0]
                self.state[ch,0] = w0
        return np.clip(y, -1.0, 1.0)



# Constants
MAX_CONCURRENT_SOUNDS = 64

# Globals
sounds: Dict[str, tuple] = {}
active_streams: List[StreamInfo] = []
streams_lock = threading.Lock()
active_count = threading.Event()
active_count.set()
mic_stream = None
mic_lock = threading.Lock()

def _resample_audio(data: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    """Resamples audio data to a new sample rate.

    Uses linear interpolation to resample the audio data from the source sample rate to the target sample rate.

    Args:
        data (np.ndarray): The audio data to resample.
        src_rate (int): The original sample rate of the audio data.
        target_rate (int): The desired sample rate.

    Returns:
        np.ndarray: The resampled audio data.

    """
    if src_rate == target_rate:
        return data

    # Calculate new length
    ratio = target_rate / src_rate
    new_len = int(len(data) * ratio)

    # Create time arrays
    old_time = np.linspace(0, 1, len(data))
    new_time = np.linspace(0, 1, new_len)

    # Handle mono vs stereo
    if len(data.shape) == 1:
        return np.interp(new_time, old_time, data)

    # Handle stereo by resampling each channel
    resampled = np.zeros((new_len, data.shape[1]), dtype=np.float32)
    for channel in range(data.shape[1]):
        resampled[:, channel] = np.interp(new_time, old_time, data[:, channel])

    return resampled

def _make_cache_key(filename: str, volume: float, normalize_db: int, bass_boost: float) -> str:
    """Creates a unique cache key for audio processing parameters.

    Combines the filename and audio processing parameters into a string to be used as a cache key.

    Args:
        filename (str): The name of the audio file.
        volume (float): The volume adjustment.
        normalize_db (int): The normalization level in dB.
        bass_boost (float): The bass boost level.

    Returns:
        str: The generated cache key.

    """
    return f"{filename}|{volume}|{normalize_db}|{bass_boost}"

def loadSound(filename: str, volume: float = 1.0, normalize_db: int = None, bass_boost: float = 0.0, silence_threshold: float = 0.01) -> tuple:
    """Loads and processes an audio file.

    Loads the audio, applies optional normalization, bass boost, and volume adjustment, and caches the result.

    Args:
        filename (str): Path to the audio file.
        volume (float, optional): Volume adjustment (0.0 to 1.0). Defaults to 1.0.
        normalize_db (int, optional): Target loudness in dB. If None, no normalization is applied. Defaults to None.
        bass_boost (float, optional): Bass boost level. Defaults to 0.0.
        silence_threshold (float, optional): Threshold for silence detection. Defaults to 0.01.

    Returns:
        tuple: A tuple containing the processed audio data and the sample rate.

    Raises:
        RuntimeError: If the audio file cannot be loaded or processed.

    """
    try:
        cache_key = _make_cache_key(filename, volume, normalize_db, bass_boost)

        if cache_key not in sounds:
            # Load raw audio if not cached
            if filename not in sounds:
                data, samplerate = sf.read(filename)

                # Remove silence from start
                if len(data.shape) == 1:
                    start_idx = np.where(np.abs(data) > silence_threshold)[0]
                else:
                    start_idx = np.where(np.max(np.abs(data), axis=1) > silence_threshold)[0]
                if len(start_idx) > 0:
                    data = data[start_idx[0]:]

                sounds[filename] = (data, samplerate)  # Cache raw audio
            else:
                data, samplerate = sounds[filename]  # Get cached raw audio

            # Apply processing
            processed_data = data.copy()

            if normalize_db is not None:
                rms = np.sqrt(np.mean(processed_data**2))
                current_db = 20 * np.log10(rms) if rms > 0 else -100
                gain_db = normalize_db - current_db
                gain_linear = 10 ** (gain_db / 20)
                processed_data = processed_data * gain_linear

            if bass_boost > 0:
                audio_filter = IIRFilter(samplerate, bass_boost)
                processed_data = audio_filter.process(processed_data)

            processed_data = processed_data * volume
            processed_data = np.clip(processed_data, -1.0, 1.0)

            sounds[cache_key] = (processed_data, samplerate)  # Cache processed version

        return sounds[cache_key]
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {filename}: {str(e)}") from e

def playSound(filename: str, device: str = None, volume: float = 1.0, normalize_db:int = -20, bass_boost: float = 0.0) -> None:
    """Play sound non-blocking with optional device selection, volume control, and bass boost

    Args:
        filename: Path to sound file
        device: Output device name/index (optional)
        volume: Volume multiplier (0.0 to 3.0, default 1.0)
        normalize: Whether to normalize audio volume (default: False)
        bass_boost: Amount of bass boost in dB (default: 0.0)
    """
    try:
        with streams_lock:
            # Check sound limit
            if len(active_streams) >= MAX_CONCURRENT_SOUNDS:
                # Remove finished streams first
                active_streams[:] = [s for s in active_streams if s.stream.active]
                # If still too many, stop oldest stream
                if len(active_streams) >= MAX_CONCURRENT_SOUNDS:
                    oldest = min(active_streams, key=lambda x: x.timestamp)
                    oldest.stream.abort()
                    oldest.stream.close()
                    active_streams.remove(oldest)

        # Load and process audio
        data, src_samplerate = loadSound(filename, volume, normalize_db, bass_boost)

        device_info = sd.query_devices(device=device or sd.default.device[1])
        target_samplerate = int(device_info['default_samplerate'])
        data = _resample_audio(data, src_samplerate, target_samplerate)

        if len(data.shape) == 1:
            data = np.column_stack((data, data))
        data = data.astype(np.float32) * np.float32(volume)

        # Create filter if bass boost enabled
        audio_filter = IIRFilter(target_samplerate, bass_boost) if bass_boost > 0 else None

        def _audio_callback(outdata, frames, time, status):
            if status:
                print(f'Status: {status}')

            remaining = len(data) - stream.current_frame
            if remaining == 0:
                raise sd.CallbackStop()

            valid_frames = min(remaining, frames)
            chunk = data[stream.current_frame:stream.current_frame + valid_frames]

            # Apply bass boost in realtime if enabled
            if audio_filter is not None:
                chunk = audio_filter.process(chunk)

            outdata[:valid_frames] = chunk
            if valid_frames < frames:
                outdata[valid_frames:] = 0
            stream.current_frame += valid_frames

        stream = sd.OutputStream(
            samplerate=target_samplerate,
            channels=data.shape[1],
            callback=_audio_callback,
            device=device
        )

        stream.current_frame = 0
        stream_info = StreamInfo(stream, time.time())

        with streams_lock:
            active_streams.append(stream_info)

        stream.start()

        def _cleanup():
            try:
                while stream.active:
                    sd.sleep(100)

                with streams_lock:
                    if stream_info in active_streams:
                        try:
                            if stream.active:  # Double check inside lock
                                stream.abort()
                            stream.close()
                            active_streams.remove(stream_info)
                        except Exception:
                            # Ignore errors during cleanup
                            ...
            except Exception:
                # Ignore any errors in cleanup thread
                ...

        cleanup_thread = threading.Thread(target=_cleanup, daemon=True)
        cleanup_thread.start()

    except Exception as e:
        raise RuntimeError(f"Failed to play sound {filename}: {str(e)}")

def stopAll() -> None:
    """Stops all currently playing audio streams.

    Aborts and closes all active audio streams, effectively stopping all playback.
    """
    global active_streams
    with streams_lock:
        for stream_info in active_streams:
            try:
                stream_info.stream.abort()
                stream_info.stream.close()
            except Exception:
                ...  # Ignore errors during forced cleanup
        active_streams.clear()

def startMicPassthrough(output_device, input_device=None, volume: float = 1.0) -> None:
    """Start microphone passthrough with resampling support

    Args:
        output_device: Output device name or index (required)
        input_device: Input device name or index (default: system default)
        volume: Volume multiplier (0.0 to 2.0, default: 1.0)
    """
    global mic_stream

    try:
        # Use system default for input if not specified
        if input_device is None:
            input_device = sd.default.device[0]

        # Get device info and sample rates
        input_info = sd.query_devices(input_device)
        output_info = sd.query_devices(output_device)

        input_rate = int(input_info['default_samplerate'])
        output_rate = int(output_info['default_samplerate'])

        with mic_lock:
            if mic_stream is not None:
                stopMicPassthrough()

            def callback(indata, outdata, frames, time, status):
                # Ensure correct channel count
                indata = indata[:, :2] if indata.shape[1] > 2 else indata


                # Resample if rates differ
                if input_rate != output_rate:
                    resampled = _resample_audio(indata, input_rate, output_rate)
                    frames_to_write = min(len(resampled), len(outdata))
                    outdata[:frames_to_write] = resampled[:frames_to_write] * volume
                    if frames_to_write < len(outdata):
                        outdata[frames_to_write:] = 0
                else:
                    outdata[:] = indata * volume

            mic_stream = sd.Stream(
                device=(input_device, output_device),
                callback=callback,
                dtype=np.float32,
                samplerate=output_rate,
                channels=(input_info['max_input_channels'],
                         output_info['max_output_channels'])
            )
            mic_stream.start()

    except Exception as e:
        print(f"\nError details: {str(e)}")
        raise RuntimeError(f"Failed to start mic passthrough: {str(e)}") from e

def stopMicPassthrough() -> None:
    """Stops the microphone passthrough stream.

    If a microphone passthrough stream is active, this function aborts and closes it.

    Returns:
        None

    """
    global mic_stream

    with mic_lock:
        if mic_stream is not None:
            mic_stream.abort()
            mic_stream.close()
            mic_stream = None


# Update __all__
__all__ = [loadSound, playSound, stopAll, startMicPassthrough, stopMicPassthrough]
