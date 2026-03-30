import numpy as np
import resampy
import torch
import torchcrepe

__all__ = ['CENTS_PER_BIN',
           'MAX_FMAX',
           'PITCH_BINS',
           'SAMPLE_RATE',
           'WINDOW_SIZE',
           'UNVOICED',
           'maximally_parallel_predict',
           'resample']

CENTS_PER_BIN = 20  # cents
MAX_FMAX = 2006.  # hz
PITCH_BINS = 360
SAMPLE_RATE = 16000  # hz
WINDOW_SIZE = 1024  # samples
UNVOICED = np.nan


def resample(audio, sample_rate):
    device = audio.device
    audio = audio.detach().cpu().numpy().squeeze(0)
    audio = resampy.resample(audio, sample_rate, SAMPLE_RATE)
    return torch.tensor(audio, device=device).unsqueeze(0)


def _unfold_chunk(padded, start_frame, num_frames, hop_length):
    start_sample = start_frame * hop_length
    end_sample = start_sample + (num_frames - 1) * hop_length + WINDOW_SIZE

    # Extract just the audio needed for this chunk
    chunk_audio = padded[:, start_sample:end_sample]

    # Unfold only this small section
    frames = torch.nn.functional.unfold(
        chunk_audio[:, None, None, :],
        kernel_size=(1, WINDOW_SIZE),
        stride=(1, hop_length)
    )
    frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE)

    # Normalize
    frames = frames - frames.mean(dim=1, keepdim=True)
    frames = frames / frames.std(dim=1, keepdim=True).clamp(min=1e-10)

    return frames


def maximally_parallel_predict(audio,
                               sample_rate,
                               hop_length=None,
                               fmin=50.,
                               fmax=MAX_FMAX,
                               model='full',
                               return_periodicity=False,
                               infer_batch_size=1024,  # Frames per batch
                               device='cuda'):

    # Default hop length: 10ms
    hop_length = SAMPLE_RATE // 100 if hop_length is None else hop_length

    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        audio = resample(audio, sample_rate)
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)

    # Keep audio on CPU until needed for each chunk
    audio = audio.cpu()

    # Calculate frames
    total_frames = 1 + int(audio.size(1) // hop_length)

    # Pad once (stays on CPU)
    padded = torch.nn.functional.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))

    # Load model once
    if not hasattr(maximally_parallel_predict, 'model') or \
            not hasattr(maximally_parallel_predict, 'capacity') or \
            maximally_parallel_predict.capacity != model:
        torchcrepe.load.model(device, model)
        maximally_parallel_predict.model = torchcrepe.infer.model
        maximally_parallel_predict.capacity = model

    model_obj = maximally_parallel_predict.model.to(device).eval()

    # Process in chunks: unfold + infer for each batch independently
    all_logits = []
    with torch.no_grad():
        for i in range(0, total_frames, infer_batch_size):
            # Number of frames in this chunk
            chunk_frames = min(infer_batch_size, total_frames - i)

            # Unfold JUST this chunk (CPU)
            frames_chunk = _unfold_chunk(padded, i, chunk_frames, hop_length)

            # Move to GPU, infer, move back to CPU
            frames_chunk = frames_chunk.to(device)
            logits = model_obj(frames_chunk, embed=False)
            all_logits.append(logits.cpu())

            # Explicit cleanup
            del frames_chunk, logits

    # Concatenate all logits on CPU, then move to GPU for postprocessing
    logits = torch.cat(all_logits, dim=0).to(device)

    # Reshape to (batch=1, 360, time)
    logits = logits.reshape(1, total_frames, PITCH_BINS).transpose(1, 2)

    # Postprocess (vectorized, minimal memory)
    min_bin = int(torchcrepe.convert.frequency_to_bins(torch.tensor(fmin, device=device)))
    max_bin = int(torchcrepe.convert.frequency_to_bins(torch.tensor(fmax, device=device)))

    logits[:, :min_bin, :] = -float('inf')
    logits[:, max_bin:, :] = -float('inf')

    # Decode
    bins = logits.argmax(dim=1)
    pitch = torchcrepe.convert.bins_to_frequency(bins.float())

    if not return_periodicity:
        return pitch

    probs = torch.sigmoid(logits)
    periodicity_vals = probs.gather(1, bins.unsqueeze(1)).squeeze(1)

    return pitch, periodicity_vals


def maximally_parallel_predict_weighted(audio,
                                        sample_rate,
                                        hop_length=None,
                                        fmin=50.,
                                        fmax=MAX_FMAX,
                                        model='full',
                                        return_periodicity=False,
                                        infer_batch_size=1024,
                                        device='cuda',
                                        window=4):

    hop_length = SAMPLE_RATE // 100 if hop_length is None else hop_length

    if sample_rate != SAMPLE_RATE:
        audio = resample(audio, sample_rate)
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)

    audio = audio.cpu()
    total_frames = 1 + int(audio.size(1) // hop_length)

    padded = torch.nn.functional.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))

    if not hasattr(maximally_parallel_predict_weighted, 'model') or \
            not hasattr(maximally_parallel_predict_weighted, 'capacity') or \
            maximally_parallel_predict_weighted.capacity != model:
        torchcrepe.load.model(device, model)
        maximally_parallel_predict_weighted.model = torchcrepe.infer.model
        maximally_parallel_predict_weighted.capacity = model

    model_obj = maximally_parallel_predict_weighted.model.to(device).eval()

    all_logits = []
    with torch.no_grad():
        for i in range(0, total_frames, infer_batch_size):
            chunk_frames = min(infer_batch_size, total_frames - i)
            frames_chunk = _unfold_chunk(padded, i, chunk_frames, hop_length)

            frames_chunk = frames_chunk.to(device)
            logits = model_obj(frames_chunk, embed=False)
            all_logits.append(logits.cpu())

            del frames_chunk, logits

    logits = torch.cat(all_logits, dim=0).to(device)
    logits = logits.reshape(1, total_frames, PITCH_BINS).transpose(1, 2)

    # Frequency masking
    min_bin = int(torchcrepe.convert.frequency_to_bins(torch.tensor(fmin, device=device)))
    max_bin = int(torchcrepe.convert.frequency_to_bins(torch.tensor(fmax, device=device)))
    logits[:, :min_bin, :] = -float('inf')
    logits[:, max_bin:, :] = -float('inf')

    # Weighted argmax
    bins_idx = logits.argmax(dim=1)

    bin_range = torch.arange(PITCH_BINS, device=device).view(1, -1, 1)
    centers = bins_idx.unsqueeze(1)
    mask = (bin_range >= centers - window) & (bin_range <= centers + window)
    masked_logits = torch.where(mask, logits, torch.tensor(-float('inf'), device=device))

    probs = torch.sigmoid(masked_logits)

    if not hasattr(maximally_parallel_predict_weighted, 'cent_weights'):
        cent_weights = torchcrepe.convert.bins_to_cents(torch.arange(PITCH_BINS))
        maximally_parallel_predict_weighted.cent_weights = cent_weights

    weights = maximally_parallel_predict_weighted.cent_weights.to(device).view(1, -1, 1)
    cents = (weights * probs).sum(dim=1) / probs.sum(dim=1)
    pitch = torchcrepe.convert.cents_to_frequency(cents)

    if not return_periodicity:
        return pitch

    periodicity_vals = probs.sum(dim=1)
    return pitch, periodicity_vals