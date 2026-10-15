import os
import torch
from typing import List, Dict, Union
from config import *


def parse_vtt(vtt_path) -> List[Dict[str, Union[float, str]]]:
    '''
    Parse VTT file to extract subtitles with start/end times (in seconds) and text.
    Filters out non-subtitle entries (e.g., '[Music]').
    Returns list of dicts: {'start': float, 'end': float, 'text': str, 'duration': float}
    '''
    if not os.path.exists(vtt_path):
        raise FileNotFoundError(f'VTT file not found: {vtt_path}')
    
    subtitles = []
    try:
        if USE_WEBVTT:
            for subtitle in webvtt.read(vtt_path):
                text = subtitle.text.strip()
                if not text or text.startswith('[') and text.endswith(']'): continue # Skip non-verbal like [Music]
                # webvtt-py 0.5.x `*_in_seconds` TRUNCATES to whole integer seconds (00:00:06.160 -> 6), which quantizes every 
                # GT boundary to 1s granularity (up to 12 frames @ 12.5fps). Parse the canonical HH:MM:SS.mmm strings instead 
                # to keep millisecond precision.
                start_sec, end_sec = time_to_seconds(subtitle.start), time_to_seconds(subtitle.end)
                if start_sec >= end_sec: continue # Invalid timing
                if text: subtitles.append({'start': start_sec, 'end': end_sec, 'text': text, 'duration': end_sec - start_sec})
        else:
            i = 0
            lines = open(vtt_path, 'r').readlines()
            while i < len(lines):
                line = lines[i].strip()
                if '-->' not in line: i += 1
                else: # Timestamp line, e.g., 00:00:01.000 --> 00:00:05.000
                    # i += 1
                    # text = ''
                    # while i < len(lines) and not '-->' in lines[i] and lines[i].strip() != '':
                    #     text += lines[i].strip() + ' '
                    #     i += 1
                    
                    text = lines[i+1].strip()
                    i += 2
                    
                    if not text or text.startswith('[') and text.endswith(']'): continue # Skip non-verbal like [Music]
                    times = line.split(' --> ')
                    start_str = times[0].replace(',', '.')
                    end_str = times[1].replace(',', '.')

                    # HH:MM:SS.ms to sec
                    start_sec = sum(float(x) * 60 ** j for j, x in enumerate(reversed(start_str.split(':')))) 
                    end_sec = sum(float(x) * 60 ** j for j, x in enumerate(reversed(end_str.split(':'))))
                    if start_sec >= end_sec: continue # Invalid timing
                    if text: subtitles.append({'start': start_sec, 'end': end_sec, 'text': text, 'duration': end_sec - start_sec})
    except Exception as e:
        raise ValueError(f'Error parsing VTT file {vtt_path}: {e}')
    return sorted(subtitles, key=lambda x: x['start'])  # Ensure chronological order


def time_to_seconds(time_str): # Convert HH:MM:SS.mmm to seconds float
    if ':' not in time_str: return float(time_str)
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(time_str)


def cw_to_se(cw: torch.Tensor) -> torch.Tensor:
    '''
    Convert (center, length) to (start, end). Inputs are normalized to [0, 1].
    cw: (N, 2) or (..., 2)
    Returns: (..., 2) as (start, end)
    '''
    c = cw[..., 0]
    w = torch.clamp(cw[..., 1], min=1e-6)
    box = [c - 0.5 * w, c + 0.5 * w]
    return torch.stack(box, dim=-1)


def box_iou(pred_se: torch.Tensor, target_se: torch.Tensor) -> torch.Tensor:
    area1 = pred_se[:, 1] - pred_se[:, 0]
    area2 = target_se[:, 1] - target_se[:, 0]
    left_top = torch.max(pred_se[:, None, 0], target_se[:, 0])
    right_bottom = torch.min(pred_se[:, None, 1], target_se[:, 1]) 
    inter = (right_bottom - left_top).clamp(min=0) 
    union = area1[:, None] + area2 - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return iou, union


def generalized_box_iou(pred_se, target_se):
    # Degenerate boxes gives inf / nan results, so do an early check
    assert (pred_se[:, 1:] >= pred_se[:, :1]).all()
    assert (target_se[:, 1:] >= target_se[:, :1]).all()
    iou, union = box_iou(pred_se, target_se)
    top_left = torch.min(pred_se[:, None, 0], target_se[:, 0])
    right_bottom = torch.max(pred_se[:, None, 1], target_se[:, 1])
    area = (right_bottom - top_left).clamp(min=0)
    return torch.where(area > 0, iou - (area - union) / area, torch.zeros_like(area))