import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
from typing import Dict, List, NamedTuple

class StreamInfo(NamedTuple):
    stream: sd.OutputStream
    timestamp: float

# Constants
MAX_CONCURRENT_SOUNDS = 100

# Globals
sounds: Dict[str, tuple] = {}
active_streams: List[StreamInfo] = []
streams_lock = threading.Lock()
active_count = threading.Event()
active_count.set()

def _resample_if_needed(data: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio data if source and target rates differ using numpy linear interpolation"""
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

def loadFile(filename: str, normalize: bool = False, target_db: float = -20.0, silence_threshold: float = 0.01) -> tuple:
    """Load audio file and cache it with optional normalization and silence removal
    
    Args:
        filename: Path to sound file
        normalize: Whether to normalize audio volume (default: False)
        target_db: Target RMS level in dB for normalization (default: -20.0)
        silence_threshold: Amplitude threshold below which frames are considered silent (default: 0.01)
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    try:
        if filename not in sounds:
            data, samplerate = sf.read(filename)

            # Remove silence from start
            if len(data.shape) == 1:  # Mono
                start_idx = np.where(np.abs(data) > silence_threshold)[0]
            else:  # Stereo
                start_idx = np.where(np.max(np.abs(data), axis=1) > silence_threshold)[0]
            if len(start_idx) > 0:
                data = data[start_idx[0]:]
            if normalize:
                # Calculate current RMS level
                rms = np.sqrt(np.mean(data**2))
                current_db = 20 * np.log10(rms) if rms > 0 else -100

                # Calculate gain needed
                gain_db = target_db - current_db
                gain_linear = 10 ** (gain_db / 20)

                # Apply gain with peak limiting
                data = data * gain_linear
                data = np.clip(data, -1.0, 1.0)

            sounds[filename] = (data, samplerate)
        return sounds[filename]
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {filename}: {str(e)}") from e

def playSound(filename: str, device: str = None, volume: float = 1.0, normalize:bool = False) -> None:
    """Play sound non-blocking with optional device selection and volume control

    Args:
        filename: Path to sound file
        device: Output device name/index (optional)
        volume: Volume multiplier (0.0 to 3.0, default 1.0)
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

        data, src_samplerate = loadFile(filename,normalize=normalize)

        # Get device info and resample if needed
        device_info = sd.query_devices(device=device or sd.default.device[1])
        target_samplerate = int(device_info['default_samplerate'])
        data = _resample_if_needed(data, src_samplerate, target_samplerate)

        # Convert to float32 stereo
        if len(data.shape) == 1:
            data = np.column_stack((data, data))
        data = data.astype(np.float32)

        # Apply volume
        data = data * np.float32(volume)

        def _audio_callback(outdata, frames, time, status):
            if status:
                print(f'Status: {status}')

            remaining = len(data) - stream.current_frame
            if remaining == 0:
                raise sd.CallbackStop()

            valid_frames = min(remaining, frames)
            outdata[:valid_frames] = data[stream.current_frame:stream.current_frame + valid_frames]
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
                                stream.stop()
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
    """Stop all active sound playback"""
    with streams_lock:
        for stream_info in active_streams:
            try:
                stream_info.stream.abort()
                stream_info.stream.close()
            except Exception:
                ...  # Ignore errors during forced cleanup
        active_streams.clear()

__all__ = [loadFile,playSound,stopAll]
