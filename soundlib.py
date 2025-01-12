import sounddevice as sd
import soundfile as sf
import numpy as np

streams = []
sounds = {}

def loadFile(filename: str):
    sounds[filename] = sf.read(filename)

def _convertSound(data, samplerate, to_samplerate):
    if samplerate == to_samplerate:
        return data, samplerate

    duration = len(data) / samplerate

    new_length = int(duration * to_samplerate)
    if data.ndim == 1:
        data = np.interp(np.linspace(0, len(data), new_length), np.arange(len(data)), data)
        data = data.reshape(-1, 1)
    elif data.ndim == 2:
        data = np.array([np.interp(np.linspace(0, len(data), new_length), np.arange(len(data)), data[:, i]) for i in range(data.shape[1])]).T

    return data, to_samplerate

def _play(data, samplerate, device=None):
    data = data.astype('float32')
    devices = [device, None] if device else [None]
    
    for dev in devices:
        default_sr = sd.query_devices(dev, 'output')['default_samplerate']
        converted_data, converted_sr = _convertSound(data, samplerate, default_sr)

        def create_callback(stream_ref):
            local_data = converted_data.copy()
            def callback(outdata, frames, time, status):
                nonlocal local_data
                if len(local_data) >= frames:
                    outdata[:frames] = local_data[:frames]
                    local_data = local_data[frames:]
                else:
                    outdata[:len(local_data)] = local_data
                    outdata[len(local_data):] = 0
                    try:
                        if stream_ref in streams:
                            streams.remove(stream_ref)
                            stream_ref.stop()
                            stream_ref.close()
                    except:
                        pass
            return callback

        stream = sd.OutputStream(
            samplerate=converted_sr,
            device=dev,
            channels=converted_data.shape[1]
        )
        stream.callback = create_callback(stream)
        streams.append(stream)
        stream.start()

def playSound(filename: str, device=None):
    if filename not in sounds:
        loadFile(filename)

    data, sr = sounds[filename]
    _play(data, sr, device)

def stopAll():
    global streams

    while streams:
        stream = streams.pop()
        stream.abort()
        stream.close()

def mixSounds(filenames: list, device=None):
    mixed_data = None
    max_sr = 0
    max_length = 0

    for filename in filenames:
        if filename not in sounds:
            loadFile(filename)
        data, sr = sounds[filename]
        print(f"Loaded {filename}: shape={data.shape}, samplerate={sr}")
        if sr > max_sr:
            max_sr = sr
        if len(data) > max_length:
            max_length = len(data)

    for filename in filenames:
        data, sr = sounds[filename]
        data, sr = _convertSound(data, sr, max_sr)
        print(f"Converted {filename}: shape={data.shape}, samplerate={sr}")
        if len(data) < max_length:
            data = np.pad(data, ((0, max_length - len(data)), (0, 0)), 'constant')
            print(f"Padded {filename}: shape={data.shape}")

        if mixed_data is None:
            mixed_data = np.zeros_like(data)
        mixed_data += data

    mixed_data = np.clip(mixed_data, -1, 1)  # Prevent clipping
    print(f"Mixed data: shape={mixed_data.shape}, samplerate={max_sr}")
    _play(mixed_data, max_sr, device)