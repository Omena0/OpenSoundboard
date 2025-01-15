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
    def __init__(self, samplerate: int, boost: float = 1.5):
        # Pre-calculate coefficients
        f = 150.0
        w0 = 2 * np.pi * (f / samplerate)
        cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
        alpha = sin_w0 / 2
        A = np.power(10, boost/40)
        
        # Optimized coefficient calculation
        self.b = np.array([
            A * ((A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha),
            2*A * ((A-1) - (A+1)*cos_w0),
            A * ((A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
        ], dtype=np.float32)
        
        a0 = (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
        self.a = np.array([
            1.0,
            -2 * ((A-1) + (A+1)*cos_w0) / a0,
            ((A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha) / a0
        ], dtype=np.float32)
        
        self.b /= a0
        self.state = np.zeros((2, 2), dtype=np.float32)
        self.buffer = None

    def process(self, x):
        if self.buffer is None or self.buffer.shape != x.shape:
            self.buffer = np.zeros_like(x, dtype=np.float32)
        
        # Vectorized processing
        for ch in range(x.shape[1]):
            self.buffer[:,ch] = np.convolve(x[:,ch], self.b, mode='same') - \
                               np.convolve(x[:,ch], self.a[1:], mode='same')
        return np.clip(self.buffer, -1.0, 1.0)

class Resampler:
    def __init__(self, data, src_rate, target_rate):
        self.data = data
        self.ratio = target_rate / src_rate
        self.src_pos = 0
        self.src_frames = len(data)
        self.buffer = None
        self.old_time = None
        self.new_time = None
    
    def process_chunk(self, frames_needed):
        src_frames = min(int(frames_needed / self.ratio) + 2, self.src_frames - self.src_pos)
        if src_frames <= 0:
            return None

        src_chunk = self.data[self.src_pos:self.src_pos + src_frames]
        self.src_pos += src_frames
        
        # Pre-allocate buffers
        if self.buffer is None or self.buffer.shape[0] != int(src_frames * self.ratio):
            self.old_time = np.linspace(0, 1, src_frames, dtype=np.float32)
            self.new_time = np.linspace(0, 1, int(src_frames * self.ratio), dtype=np.float32)
            self.buffer = np.zeros((int(src_frames * self.ratio), src_chunk.shape[1]), dtype=np.float32)
        
        # Vectorized resampling
        for ch in range(src_chunk.shape[1]):
            self.buffer[:, ch] = np.interp(self.new_time, self.old_time, src_chunk[:, ch])
        
        return self.buffer


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

def _make_cache_key(filename: str, volume: float, normalize_db: int, bass_boost: float) -> str:
    """Create cache key including all processing parameters"""
    return f"{filename}|{volume}|{normalize_db}|{bass_boost}"

def loadSound(
        filename: str,
        volume: float = 1.0,
        normalize_db: int = None,
        bass_boost: float = 0.0
    ) -> tuple:
    try:
        cache_key = _make_cache_key(filename, volume, normalize_db, bass_boost)

        if cache_key not in sounds:
            # Load raw audio if not cached
            if filename not in sounds:
                data, sr = sf.read(filename)
                sounds[filename] = (data, sr)  # Cache raw
            else:
                data, sr = sounds[filename]  # Get cached raw

            # Process audio
            processed_data = data.copy()

            # Apply normalization if needed
            if normalize_db is not None:
                rms = np.sqrt(np.mean(processed_data**2))
                current_db = 20 * np.log10(rms) if rms > 0 else -100
                gain_db = normalize_db - current_db
                gain_linear = 10 ** (gain_db / 20)
                processed_data = processed_data * gain_linear

            # Apply bass boost if needed
            if bass_boost > 0:
                audio_filter = IIRFilter(sr, bass_boost)
                processed_data = audio_filter.process(processed_data)

            # Apply volume and clip
            processed_data = processed_data * volume
            processed_data = np.clip(processed_data, -1.0, 1.0)

            sounds[cache_key] = (processed_data, sr)

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
        # Check sound limit
        with streams_lock:
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

        # Create resampling state
        resampler = Resampler(data, src_samplerate, target_samplerate)

        if len(data.shape) == 1:
            data = np.column_stack((data, data))
        data = data.astype(np.float32) * np.float32(volume)

        # Create filter if bass boost enabled
        audio_filter = IIRFilter(target_samplerate, bass_boost) if bass_boost > 0 else None

        def _audio_callback(outdata, frames, time, status):
            if status:
                print(f'Error: {status}')

            # Resample chunk
            chunk = resampler.process_chunk(frames)
            if chunk is None:
                raise sd.CallbackStop()

            # Apply bass boost in realtime if enabled
            if audio_filter is not None:
                chunk = audio_filter.process(chunk)

            # Copy resampled data to output
            valid_frames = min(len(chunk), frames)
            outdata[:valid_frames] = chunk[:valid_frames]
            if valid_frames < frames:
                outdata[valid_frames:] = 0

        # Create output stream with larger buffer
        stream = sd.OutputStream(
            device=device,
            channels=2,
            callback=_audio_callback,
            samplerate=target_samplerate,
            dtype=np.float32,
            latency='low'
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
    """Start microphone passthrough with resampling.

    Args:
        output_device: Output device name or index (required)
        input_device: Input device name or index (default: system default)
        volume: Volume multiplier (0.0 to 2.0, default: 1.0)
    """
    global mic_stream
    try:
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

            # Create resampler if rates differ
            resampler = Resampler(None, input_rate, output_rate) if input_rate != output_rate else None

            def callback(indata, outdata, frames, time, status):
                if status:
                    print(f'Status: {status}')

                # Ensure correct channel count
                audio = indata[:, :2] if indata.shape[1] > 2 else indata

                # Resample if needed
                if resampler is not None:
                    resampler.data = audio  # Update source data
                    audio = resampler.process_chunk(frames)
                    if audio is None:
                        outdata.fill(0)
                        return

                # Copy to output with volume
                valid_frames = min(len(audio), frames)
                outdata[:valid_frames] = audio[:valid_frames] * volume
                if valid_frames < frames:
                    outdata[valid_frames:].fill(0)

            mic_stream = sd.Stream(
                device=(input_device, output_device),
                callback=callback,
                dtype=np.float32,
                samplerate=output_rate
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
__all__ = ['loadSound', 'playSound', 'stopAll', 'startMicPassthrough', 'stopMicPassthrough']
