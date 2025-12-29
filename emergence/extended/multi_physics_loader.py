#!/usr/bin/env python3
"""
================================================================================
MULTI-PHYSICS DATA LOADER

Loads real physics data from multiple domains for consciousness substrate testing.

SOURCES:
  - Hydrosphere: NDBC ocean buoy (wave height, period, wind, pressure, temp)
  - Magnetosphere: NOAA magnetometer (Bx, By, Bz, total field)
  - Heliosphere: ACE/DSCOVR solar wind (speed, density, temperature, IMF)
  - Geosphere: IRIS seismic (requires obspy)
  - Spacetime: LIGO auxiliary channels (requires h5py + local files)

USAGE:
  from multi_physics_loader import load_ocean_buoy, load_magnetometer, load_solar_wind
  
  data, meta = load_ocean_buoy(station="46221", hours=48)
  # data: (T, n_channels) array
  # meta: dict with channel names, source info

================================================================================
"""

import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    HAS_OBSPY = True
except ImportError:
    HAS_OBSPY = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# =============================================================================
# HYDROSPHERE: NDBC Ocean Buoy
# =============================================================================

def load_ocean_buoy(station: str = "46025", hours: int = 48, 
                    resample_minutes: int = 10) -> Tuple[np.ndarray, Dict]:
    """
    Load ocean buoy data from NDBC.
    
    Station 46221 is Santa Monica Bay - good multi-scale structure.
    Other options: 46025 (Santa Monica), 46222 (San Pedro), 46086 (San Clemente)
    
    Channels (typically 10):
      - WVHT: Significant wave height (m)
      - DPD: Dominant wave period (s)
      - APD: Average wave period (s)
      - MWD: Mean wave direction (deg)
      - WSPD: Wind speed (m/s)
      - GST: Wind gust (m/s)
      - WDIR: Wind direction (deg)
      - PRES: Atmospheric pressure (hPa)
      - ATMP: Air temperature (C)
      - WTMP: Water temperature (C)
    
    Returns:
        data: (T, n_channels) normalized array
        meta: dict with channel_names, station, source_url
    """
    if not HAS_REQUESTS:
        print("  requests not available, generating synthetic ocean data")
        return _generate_synthetic_ocean(hours, resample_minutes)
    
    # NDBC real-time data URL
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
    except Exception as e:
        print(f"  Failed to fetch NDBC data: {e}")
        print("  Generating synthetic ocean data")
        return _generate_synthetic_ocean(hours, resample_minutes)
    
    # Parse header
    header = lines[0].replace('#', '').split()
    units = lines[1].replace('#', '').split()
    
    # Channels we want
    target_channels = ['WVHT', 'DPD', 'APD', 'MWD', 'WSPD', 'GST', 'WDIR', 'PRES', 'ATMP', 'WTMP']
    channel_indices = []
    channel_names = []
    
    for ch in target_channels:
        if ch in header:
            channel_indices.append(header.index(ch))
            channel_names.append(ch)
    
    if len(channel_indices) < 5:
        print(f"  Only found {len(channel_indices)} channels, generating synthetic")
        return _generate_synthetic_ocean(hours, resample_minutes)
    
    # Parse data lines
    data_rows = []
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < max(channel_indices) + 1:
            continue
        
        try:
            y = parts[0]
            if y in ("MM", "NA"):
                continue
            year = int(y)
            
            # NDBC realtime2 uses #YY (two-digit year)
            if year < 100:
                year += 2000
            
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            timestamp = datetime(year, month, day, hour, minute)
            
            if timestamp < cutoff_time:
                continue
            
            row = []
            valid = True
            for idx in channel_indices:
                val = parts[idx]
                if val == 'MM' or val == 'NA':
                    valid = False
                    break
                row.append(float(val))
            
            if valid:
                data_rows.append(row)
        except (ValueError, IndexError):
            continue
    
    if len(data_rows) < 10:
        print(f"  Only {len(data_rows)} valid rows, generating synthetic")
        return _generate_synthetic_ocean(hours, resample_minutes)
    
    # Convert to array (most recent last)
    data = np.array(data_rows[::-1])
    
    # Normalize each channel to [-1, 1]
    data_norm = np.zeros_like(data)
    for i in range(data.shape[1]):
        col = data[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-10:
            data_norm[:, i] = 2 * (col - col_min) / (col_max - col_min) - 1
        else:
            data_norm[:, i] = 0
    
    meta = {
        "channel_names": channel_names,
        "station": station,
        "source": "NDBC",
        "url": url,
        "n_samples": len(data_norm),
        "n_channels": len(channel_names),
    }
    
    print(f"  Loaded {meta['n_samples']} samples, {meta['n_channels']} channels from NDBC {station}")
    
    return data_norm, meta


def _generate_synthetic_ocean(hours: int, resample_minutes: int) -> Tuple[np.ndarray, Dict]:
    """Generate synthetic ocean-like data with multi-scale structure."""
    rng = np.random.default_rng(42)
    n_samples = (hours * 60) // resample_minutes
    t = np.linspace(0, hours, n_samples)
    
    # 10 channels with different timescales
    data = np.zeros((n_samples, 10))
    
    # Wave height: tide + swell + wind waves
    data[:, 0] = (0.3 * np.sin(2*np.pi*t/12.42) +  # Semi-diurnal tide
                  0.2 * np.sin(2*np.pi*t/24) +      # Diurnal
                  0.1 * np.sin(2*np.pi*t/6) +       # Swell
                  0.1 * rng.normal(0, 1, n_samples)) # Wind waves
    
    # Wave period: correlated with height
    data[:, 1] = 0.7 * data[:, 0] + 0.3 * rng.normal(0, 0.5, n_samples)
    
    # Average period: smoothed dominant
    data[:, 2] = np.convolve(data[:, 1], np.ones(5)/5, mode='same')
    
    # Wave direction: slow variation
    data[:, 3] = 0.5 * np.sin(2*np.pi*t/48) + 0.2 * rng.normal(0, 1, n_samples)
    
    # Wind speed: weather regime
    regime = np.sin(2*np.pi*t/36)  # 36-hour weather pattern
    data[:, 4] = 0.4 * regime + 0.3 * rng.normal(0, 1, n_samples)
    
    # Wind gust: spiky version of wind
    data[:, 5] = data[:, 4] + 0.3 * np.abs(rng.normal(0, 1, n_samples))
    
    # Wind direction: correlated with wave direction
    data[:, 6] = 0.6 * data[:, 3] + 0.4 * rng.normal(0, 0.5, n_samples)
    
    # Pressure: slow, anticorrelated with wind
    data[:, 7] = -0.5 * regime + 0.1 * rng.normal(0, 1, n_samples)
    
    # Air temp: diurnal + trend
    data[:, 8] = 0.4 * np.sin(2*np.pi*t/24) + 0.1 * rng.normal(0, 1, n_samples)
    
    # Water temp: very slow, lag air
    data[:, 9] = 0.3 * np.sin(2*np.pi*(t-3)/24) + 0.05 * rng.normal(0, 1, n_samples)
    
    # Normalize
    for i in range(10):
        col = data[:, i]
        data[:, i] = (col - col.mean()) / (col.std() + 1e-10)
        data[:, i] = np.clip(data[:, i], -3, 3) / 3  # Scale to [-1, 1]
    
    meta = {
        "channel_names": ['WVHT', 'DPD', 'APD', 'MWD', 'WSPD', 'GST', 'WDIR', 'PRES', 'ATMP', 'WTMP'],
        "station": "SYNTHETIC",
        "source": "synthetic",
        "n_samples": n_samples,
        "n_channels": 10,
    }
    
    print(f"  Generated {n_samples} synthetic ocean samples, 10 channels")
    return data, meta


# =============================================================================
# MAGNETOSPHERE: NOAA Magnetometer
# =============================================================================

def load_magnetometer(hours: int = 48) -> Tuple[np.ndarray, Dict]:
    """
    Load magnetometer data from NOAA SWPC.
    
    Channels (6):
      - Bx, By, Bz: Magnetic field components (nT)
      - Bt: Total field magnitude
      - Lat, Lon: GSM coordinates (if available)
    
    Returns:
        data: (T, n_channels) normalized array
        meta: dict with channel_names, source
    """
    if not HAS_REQUESTS:
        print("  requests not available, generating synthetic magnetometer data")
        return _generate_synthetic_mag(hours)
    
    # NOAA SWPC real-time magnetometer
    url = "https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        json_data = response.json()
    except Exception as e:
        print(f"  Failed to fetch magnetometer data: {e}")
        return _generate_synthetic_mag(hours)
    
    # Parse JSON (first row is header)
    header = json_data[0]
    rows = json_data[1:]
    
    # Extract Bx, By, Bz, Bt
    channel_map = {'bx_gsm': 'Bx', 'by_gsm': 'By', 'bz_gsm': 'Bz', 'bt': 'Bt'}
    channel_indices = {}
    channel_names = []
    
    for i, h in enumerate(header):
        h_lower = h.lower()
        if h_lower in channel_map:
            channel_indices[h_lower] = i
            channel_names.append(channel_map[h_lower])
    
    if len(channel_indices) < 3:
        print(f"  Only found {len(channel_indices)} mag channels, generating synthetic")
        return _generate_synthetic_mag(hours)
    
    # Extract data
    data_rows = []
    for row in rows:
        try:
            vals = []
            for key in ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']:
                if key in channel_indices:
                    val = row[channel_indices[key]]
                    if val is None:
                        vals.append(np.nan)
                    else:
                        vals.append(float(val))
            data_rows.append(vals)
        except (ValueError, IndexError):
            continue
    
    if len(data_rows) < 10:
        print(f"  Only {len(data_rows)} mag rows, generating synthetic")
        return _generate_synthetic_mag(hours)
    
    data = np.array(data_rows)
    
    # Handle NaN by interpolation
    for i in range(data.shape[1]):
        col = data[:, i]
        mask = np.isnan(col)
        if mask.all():
            col[:] = 0
        elif mask.any():
            col[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), col[~mask])
    
    # Normalize
    data_norm = np.zeros_like(data)
    for i in range(data.shape[1]):
        col = data[:, i]
        col_std = col.std()
        if col_std > 1e-10:
            data_norm[:, i] = (col - col.mean()) / col_std
            data_norm[:, i] = np.clip(data_norm[:, i], -3, 3) / 3
        else:
            data_norm[:, i] = 0
    
    meta = {
        "channel_names": channel_names,
        "source": "NOAA_SWPC",
        "url": url,
        "n_samples": len(data_norm),
        "n_channels": len(channel_names),
    }
    
    print(f"  Loaded {meta['n_samples']} magnetometer samples, {meta['n_channels']} channels")
    return data_norm, meta


def _generate_synthetic_mag(hours: int) -> Tuple[np.ndarray, Dict]:
    """Generate synthetic magnetometer data."""
    rng = np.random.default_rng(43)
    n_samples = hours * 60  # 1-minute resolution
    t = np.linspace(0, hours, n_samples)
    
    data = np.zeros((n_samples, 6))
    
    # Slow baseline variation
    baseline = 0.3 * np.sin(2*np.pi*t/24)
    
    # Bx, By, Bz with different phases
    data[:, 0] = baseline + 0.2 * np.sin(2*np.pi*t/6) + 0.1 * rng.normal(0, 1, n_samples)
    data[:, 1] = 0.8 * baseline + 0.3 * np.sin(2*np.pi*t/8 + 1) + 0.1 * rng.normal(0, 1, n_samples)
    data[:, 2] = 0.6 * baseline + 0.2 * np.sin(2*np.pi*t/12 + 2) + 0.15 * rng.normal(0, 1, n_samples)
    
    # Bt: magnitude
    data[:, 3] = np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)
    
    # Lat, Lon (slow drift)
    data[:, 4] = 0.1 * np.sin(2*np.pi*t/48) + 0.02 * rng.normal(0, 1, n_samples)
    data[:, 5] = 0.1 * np.cos(2*np.pi*t/48) + 0.02 * rng.normal(0, 1, n_samples)
    
    # Normalize
    for i in range(6):
        col = data[:, i]
        data[:, i] = (col - col.mean()) / (col.std() + 1e-10)
        data[:, i] = np.clip(data[:, i], -3, 3) / 3
    
    meta = {
        "channel_names": ['Bx', 'By', 'Bz', 'Bt', 'Lat', 'Lon'],
        "source": "synthetic",
        "n_samples": n_samples,
        "n_channels": 6,
    }
    
    print(f"  Generated {n_samples} synthetic magnetometer samples, 6 channels")
    return data, meta


# =============================================================================
# HELIOSPHERE: Solar Wind
# =============================================================================

def load_solar_wind(hours: int = 48) -> Tuple[np.ndarray, Dict]:
    """
    Load solar wind data from NOAA SWPC (DSCOVR/ACE).
    
    Channels (7):
      - Speed: Solar wind speed (km/s)
      - Density: Proton density (p/cc)
      - Temperature: Proton temperature (K)
      - Bx, By, Bz: IMF components (nT)
      - Bt: Total IMF
    
    Returns:
        data: (T, n_channels) normalized array
        meta: dict with channel_names, source
    """
    if not HAS_REQUESTS:
        print("  requests not available, generating synthetic solar wind data")
        return _generate_synthetic_solar(hours)
    
    # NOAA SWPC plasma data
    url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        plasma_data = response.json()
    except Exception as e:
        print(f"  Failed to fetch solar wind data: {e}")
        return _generate_synthetic_solar(hours)
    
    # Parse
    header = plasma_data[0]
    rows = plasma_data[1:]
    
    # Find indices
    speed_idx = header.index('speed') if 'speed' in header else None
    density_idx = header.index('density') if 'density' in header else None
    temp_idx = header.index('temperature') if 'temperature' in header else None
    
    if speed_idx is None:
        print("  No speed in solar wind data, generating synthetic")
        return _generate_synthetic_solar(hours)
    
    # Extract
    data_rows = []
    for row in rows:
        try:
            vals = []
            for idx in [speed_idx, density_idx, temp_idx]:
                if idx is not None and row[idx] is not None:
                    vals.append(float(row[idx]))
                else:
                    vals.append(np.nan)
            data_rows.append(vals)
        except (ValueError, IndexError):
            continue
    
    if len(data_rows) < 10:
        return _generate_synthetic_solar(hours)
    
    # Also get mag data for IMF
    mag_data, mag_meta = load_magnetometer(hours)
    
    # Combine
    plasma = np.array(data_rows)
    
    # Handle NaN
    for i in range(plasma.shape[1]):
        col = plasma[:, i]
        mask = np.isnan(col)
        if mask.all():
            col[:] = 0
        elif mask.any():
            col[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), col[~mask])
    
    # Normalize plasma
    for i in range(plasma.shape[1]):
        col = plasma[:, i]
        col_std = col.std()
        if col_std > 1e-10:
            plasma[:, i] = (col - col.mean()) / col_std
            plasma[:, i] = np.clip(plasma[:, i], -3, 3) / 3
    
    # Align lengths
    min_len = min(len(plasma), len(mag_data))
    combined = np.zeros((min_len, 7))
    combined[:, :3] = plasma[:min_len]
    combined[:, 3:7] = mag_data[:min_len, :4]  # Bx, By, Bz, Bt
    
    meta = {
        "channel_names": ['Speed', 'Density', 'Temp', 'Bx', 'By', 'Bz', 'Bt'],
        "source": "NOAA_SWPC",
        "n_samples": min_len,
        "n_channels": 7,
    }
    
    print(f"  Loaded {meta['n_samples']} solar wind samples, {meta['n_channels']} channels")
    return combined, meta


def _generate_synthetic_solar(hours: int) -> Tuple[np.ndarray, Dict]:
    """Generate synthetic solar wind data."""
    rng = np.random.default_rng(44)
    n_samples = hours * 60
    t = np.linspace(0, hours, n_samples)
    
    data = np.zeros((n_samples, 7))
    
    # Solar wind speed: slow variation + sudden jumps (CME-like)
    base_speed = 0.3 * np.sin(2*np.pi*t/27)  # ~27-day solar rotation
    jumps = np.zeros(n_samples)
    for _ in range(hours // 24):  # Occasional events
        if rng.random() < 0.3:
            idx = rng.integers(0, n_samples)
            jumps[idx:min(idx+60, n_samples)] = rng.uniform(0.3, 0.6)
    data[:, 0] = base_speed + jumps + 0.1 * rng.normal(0, 1, n_samples)
    
    # Density: anticorrelated with speed
    data[:, 1] = -0.5 * data[:, 0] + 0.2 * rng.normal(0, 1, n_samples)
    
    # Temperature: correlated with speed
    data[:, 2] = 0.6 * data[:, 0] + 0.15 * rng.normal(0, 1, n_samples)
    
    # IMF components
    data[:, 3] = 0.2 * np.sin(2*np.pi*t/12) + 0.1 * rng.normal(0, 1, n_samples)
    data[:, 4] = 0.2 * np.sin(2*np.pi*t/12 + 1) + 0.1 * rng.normal(0, 1, n_samples)
    data[:, 5] = 0.3 * np.sin(2*np.pi*t/6) + 0.15 * rng.normal(0, 1, n_samples)
    data[:, 6] = np.sqrt(data[:, 3]**2 + data[:, 4]**2 + data[:, 5]**2)
    
    # Normalize
    for i in range(7):
        col = data[:, i]
        data[:, i] = (col - col.mean()) / (col.std() + 1e-10)
        data[:, i] = np.clip(data[:, i], -3, 3) / 3
    
    meta = {
        "channel_names": ['Speed', 'Density', 'Temp', 'Bx', 'By', 'Bz', 'Bt'],
        "source": "synthetic",
        "n_samples": n_samples,
        "n_channels": 7,
    }
    
    print(f"  Generated {n_samples} synthetic solar wind samples, 7 channels")
    return data, meta


# =============================================================================
# GEOSPHERE: Seismic (requires obspy)
# =============================================================================

def load_seismic(network: str = "IU", station: str = "ANMO",
                 hours: int = 1) -> Tuple[np.ndarray, Dict]:
    """
    Load seismic data from IRIS/FDSN.
    
    Channels (2-3):
      - BHZ: Vertical broadband
      - BHN: North-South
      - BHE: East-West
    
    Returns:
        data: (T, n_channels) normalized array
        meta: dict with channel_names, station
    """
    if not HAS_OBSPY:
        print("  obspy not available, generating synthetic seismic data")
        return _generate_synthetic_seismic(hours)
    
    try:
        client = Client("IRIS")
        t2 = UTCDateTime.now()
        t1 = t2 - hours * 3600
        
        st = client.get_waveforms(network, station, "*", "BHZ,BHN,BHE", t1, t2)
        
        if len(st) == 0:
            print(f"  No seismic data from {station}, generating synthetic")
            return _generate_synthetic_seismic(hours)
        
        # Extract and align
        channels = []
        channel_names = []
        min_len = None
        
        for tr in st:
            channels.append(tr.data)
            channel_names.append(tr.stats.channel)
            if min_len is None or len(tr.data) < min_len:
                min_len = len(tr.data)
        
        # Align and stack
        data = np.zeros((min_len, len(channels)))
        for i, ch in enumerate(channels):
            data[:, i] = ch[:min_len]
        
        # Normalize
        for i in range(data.shape[1]):
            col = data[:, i]
            col_std = col.std()
            if col_std > 1e-10:
                data[:, i] = (col - col.mean()) / col_std
                data[:, i] = np.clip(data[:, i], -3, 3) / 3
        
        meta = {
            "channel_names": channel_names,
            "station": station,
            "network": network,
            "source": "IRIS",
            "n_samples": min_len,
            "n_channels": len(channel_names),
        }
        
        print(f"  Loaded {meta['n_samples']} seismic samples, {meta['n_channels']} channels")
        return data, meta
        
    except Exception as e:
        print(f"  Seismic fetch error: {e}")
        return _generate_synthetic_seismic(hours)


def _generate_synthetic_seismic(hours: int) -> Tuple[np.ndarray, Dict]:
    """Generate synthetic seismic data (highly correlated)."""
    rng = np.random.default_rng(45)
    n_samples = hours * 3600 * 20  # 20 Hz
    
    # Common signal (microseism)
    common = np.cumsum(rng.normal(0, 0.01, n_samples))
    common = common - np.mean(common)
    
    data = np.zeros((n_samples, 2))
    data[:, 0] = common + 0.05 * rng.normal(0, 1, n_samples)
    data[:, 1] = 0.98 * common + 0.07 * rng.normal(0, 1, n_samples)
    
    # Normalize
    for i in range(2):
        col = data[:, i]
        data[:, i] = (col - col.mean()) / (col.std() + 1e-10)
        data[:, i] = np.clip(data[:, i], -3, 3) / 3
    
    # Downsample
    factor = 100
    data = data[::factor]
    
    meta = {
        "channel_names": ['BHZ', 'BHN'],
        "station": "SYNTHETIC",
        "source": "synthetic",
        "n_samples": len(data),
        "n_channels": 2,
    }
    
    print(f"  Generated {len(data)} synthetic seismic samples, 2 channels")
    return data, meta


# =============================================================================
# UTILITY: Compute statistics
# =============================================================================

def compute_data_statistics(data: np.ndarray) -> Dict:
    """
    Compute autocorrelation and cross-coherence for data.
    
    Returns:
        dict with mean_autocorr, mean_crosscoh, per-channel stats
    """
    n_samples, n_channels = data.shape
    
    # Autocorrelation (lag-1) per channel
    autocorrs = []
    for i in range(n_channels):
        col = data[:, i]
        if len(col) > 1:
            r = np.corrcoef(col[:-1], col[1:])[0, 1]
            if np.isfinite(r):
                autocorrs.append(r)
    
    mean_autocorr = np.mean(autocorrs) if autocorrs else 0.0
    
    # Cross-coherence (mean pairwise correlation)
    crosscorrs = []
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            r = np.corrcoef(data[:, i], data[:, j])[0, 1]
            if np.isfinite(r):
                crosscorrs.append(abs(r))
    
    mean_crosscoh = np.mean(crosscorrs) if crosscorrs else 0.0
    
    return {
        "mean_autocorr": float(mean_autocorr),
        "mean_crosscoh": float(mean_crosscoh),
        "n_channels": n_channels,
        "n_samples": n_samples,
    }


# =============================================================================
# MAIN: Test all loaders
# =============================================================================

def main():
    print("=" * 70)
    print("MULTI-PHYSICS DATA LOADER TEST")
    print("=" * 70)
    print()
    
    results = {}
    
    print("HYDROSPHERE (Ocean Buoy):")
    ocean_data, ocean_meta = load_ocean_buoy(hours=24)
    ocean_stats = compute_data_statistics(ocean_data)
    results["hydrosphere"] = {**ocean_meta, **ocean_stats}
    print(f"  AutoCorr: {ocean_stats['mean_autocorr']:.3f}")
    print(f"  CrossCoh: {ocean_stats['mean_crosscoh']:.3f}")
    print()
    
    print("MAGNETOSPHERE (Magnetometer):")
    mag_data, mag_meta = load_magnetometer(hours=24)
    mag_stats = compute_data_statistics(mag_data)
    results["magnetosphere"] = {**mag_meta, **mag_stats}
    print(f"  AutoCorr: {mag_stats['mean_autocorr']:.3f}")
    print(f"  CrossCoh: {mag_stats['mean_crosscoh']:.3f}")
    print()
    
    print("HELIOSPHERE (Solar Wind):")
    solar_data, solar_meta = load_solar_wind(hours=24)
    solar_stats = compute_data_statistics(solar_data)
    results["heliosphere"] = {**solar_meta, **solar_stats}
    print(f"  AutoCorr: {solar_stats['mean_autocorr']:.3f}")
    print(f"  CrossCoh: {solar_stats['mean_crosscoh']:.3f}")
    print()
    
    print("GEOSPHERE (Seismic):")
    seis_data, seis_meta = load_seismic(hours=1)
    seis_stats = compute_data_statistics(seis_data)
    results["geosphere"] = {**seis_meta, **seis_stats}
    print(f"  AutoCorr: {seis_stats['mean_autocorr']:.3f}")
    print(f"  CrossCoh: {seis_stats['mean_crosscoh']:.3f}")
    print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Domain':<15} {'Channels':>8} {'AutoCorr':>10} {'CrossCoh':>10}")
    print("-" * 45)
    for domain, info in results.items():
        print(f"{domain:<15} {info['n_channels']:>8} {info['mean_autocorr']:>10.3f} {info['mean_crosscoh']:>10.3f}")
    
    return results


if __name__ == "__main__":
    main()
