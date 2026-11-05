"""
YZ Bootis 시계열 측광 분석 - 로컬 실행용 스크립트
(Colab 노트북 `시계열분석프로그램1.ipynb`의 복사본을 로컬 실행용으로 변환한 버전)

실행 방법 (cmd/터미널에서):
    python "시계열분석프로그램1_로컬.py"

첫 실행 시에는 TARGET_ID/COMP_IDS가 None이라, 별 자동탐지 미리보기 이미지(preview_stars.png)만
만들고 멈춥니다. 그 이미지를 보고 대상 별/비교성 번호를 확인한 뒤 아래 [설정] 구역의
TARGET_ID / COMP_IDS를 채우고 다시 실행하면 전체 분석(측광 -> 아웃라이어 제거 -> CSV/그래프 저장)이
진행됩니다.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 화면에 창을 띄우는 대신 파일로 저장 (cmd에서 그대로 실행 가능)
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_sources, centroid_com

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11


# ============================================================
# [설정] 여기를 본인 데이터에 맞게 수정하세요
# ============================================================

# 필터별 파일이 한 폴더에 섞여있고, 파일명 끝이 -g/-r/-i 로 구분되는 구조
# (실제 확인된 구조: C:\26astro\YZBootis\result\pp\pp_YZ Bootis_20260328-0001-g.fit 등)
BASE_DIR = r'C:\26astro\YZBootis\result\pp'
FILE_PATTERN_BY_FILTER = {
    'g': '*-g.fit',
    'r': '*-r.fit',
    'i': '*-i.fit',
}
FILTER_ORDER = list(FILE_PATTERN_BY_FILTER.keys())

OUTPUT_DIR = r'C:\26astro\YZBootis\result'  # 결과 CSV/그래프 저장 위치
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'photometry_results.csv')

# 측광 조리개 설정 (px) -- "(옵션) 조리개 반지름 자동 최적화" 단계에서 실제 데이터로 자동 갱신됨
APERTURE_RADIUS = 12
ANNULUS_R_IN = 16
ANNULUS_R_OUT = 22

CENTROID_SEARCH_BOX = 9

DATE_OBS_KEYS = ['DATE-OBS', 'DATE_OBS', 'DATEOBS']
TIME_OBS_KEYS = ['TIME-OBS', 'TIME_OBS', 'UT']

# flat 프레임으로 실측한 카메라 gain (헤더의 EGAIN=0.0495는 14배 넘게 틀려서 신뢰 불가로 확인됨).
# 아래 FLAT_FILE_1/2가 존재하면 그걸로 재측정해서 이 값을 덮어씀.
GAIN_DEFAULT = 0.71
FLAT_FILE_1 = r'C:\26astro\For Preprocess\YZBootis\A\flat\flat_g_0001.fit'
FLAT_FILE_2 = r'C:\26astro\For Preprocess\YZBootis\A\flat\flat_g_0002.fit'

# 아웃라이어 프레임 자동 제거 기준
MAX_CENTROID_SHIFT_PX = 6.0
MAX_DIFF_MAG_ERR_MULTIPLIER = 5.0
MIN_DIFF_MAG_ERR_FLOOR = 0.02

# 비교성 안정성 검증 / 조리개 최적화에 쓸 샘플 프레임 수 (속도용)
VET_SAMPLE_SIZE = 15
APERTURE_SAMPLE_SIZE = 15
COMP_VET_MAX_STD = 0.05

# 별 자동탐지 (미리보기 단계)
PREVIEW_FILTER = FILTER_ORDER[0]
DETECT_FWHM = 8.0
DETECT_THRESHOLD_SIGMA = 8.0

# 색지수 계산 시 필터 간 매칭 허용 시간차 (분)
TOLERANCE_MIN = 5.0

# 대상 별 / 비교성 번호 -- 처음 실행할 때는 모르니 None으로 둡니다.
# 1차 실행 후 저장되는 preview_stars.png를 보고 번호를 확인한 뒤 채워서 다시 실행하세요.
TARGET_ID = 0
COMP_IDS = [1, 2, 3, 4]


# ============================================================
# 공통 함수 (Colab 노트북과 동일한 로직)
# ============================================================

def list_fits_files(base_dir, pattern):
    files = sorted(glob.glob(os.path.join(base_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"'{base_dir}' 에서 '{pattern}' 패턴의 파일을 찾지 못했습니다.")
    return files


def load_fits_data(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data.astype(float), hdu.header
    raise ValueError(f"'{path}' 에서 2D 이미지 데이터를 찾지 못했습니다.")


def get_obs_time(header):
    date_val = None
    for key in DATE_OBS_KEYS:
        if key in header:
            date_val = header[key]
            break
    if date_val is None:
        raise KeyError('DATE-OBS 계열 헤더 키워드를 찾을 수 없습니다.')
    if 'T' not in str(date_val):
        for key in TIME_OBS_KEYS:
            if key in header:
                date_val = f"{date_val}T{header[key]}"
                break
    return Time(date_val, format='isot', scale='utc')


def measure_flux(data, xy, gain, aperture_radius, r_in, r_out):
    aperture = CircularAperture(xy, r=aperture_radius)
    annulus = CircularAnnulus(xy, r_in=r_in, r_out=r_out)
    ann_mask = annulus.to_mask(method='center')
    ann_data = ann_mask.multiply(data)
    ann_data_1d = ann_data[ann_mask.data > 0]
    bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(ann_data_1d)
    phot_table = aperture_photometry(data, aperture)
    raw_sum = phot_table['aperture_sum'][0]
    n_pix = aperture.area
    net_flux = raw_sum - bkg_median * n_pix
    flux_err = np.sqrt(max(net_flux, 0) / gain + n_pix * bkg_std**2)
    return net_flux, flux_err, bkg_median


def refine_centroid(data, x0, y0, box_size):
    try:
        x, y = centroid_sources(data, [x0], [y0], box_size=box_size, centroid_func=centroid_com)
        return float(x[0]), float(y[0])
    except Exception:
        return x0, y0


def instrumental_mag(flux, flux_err):
    if flux <= 0:
        return np.nan, np.nan
    mag = -2.5 * np.log10(flux)
    mag_err = 1.0857 * flux_err / flux
    return mag, mag_err


def measure_gain(file1, file2, patch_half=200):
    d1 = fits.getdata(file1).astype(float)
    d2 = fits.getdata(file2).astype(float)
    if d1.shape != d2.shape:
        raise ValueError('두 flat 프레임의 크기가 다릅니다.')
    h, w = d1.shape
    cy, cx = h // 2, w // 2
    p1 = d1[cy - patch_half:cy + patch_half, cx - patch_half:cx + patch_half]
    p2 = d2[cy - patch_half:cy + patch_half, cx - patch_half:cx + patch_half]
    m1, _, _ = sigma_clipped_stats(p1, sigma=4.0)
    m2, _, _ = sigma_clipped_stats(p2, sigma=4.0)
    _, _, diff_std = sigma_clipped_stats(p1 - p2, sigma=4.0)
    return (m1 + m2) / diff_std**2, m1, m2


def vet_comparison_stars(files, comp_xy_list, gain, aperture_radius, r_in, r_out, max_std=COMP_VET_MAX_STD):
    if len(comp_xy_list) < 3:
        print('[비교성 검증] 비교성이 3개 미만이라 안정성 검증을 건너뜁니다.')
        return comp_xy_list

    n = len(comp_xy_list)
    mag_series = [[] for _ in range(n)]
    for path in files:
        try:
            data, _ = load_fits_data(path)
            fluxes = []
            for xy in comp_xy_list:
                x, y = refine_centroid(data, *xy, box_size=CENTROID_SEARCH_BOX)
                f, _, _ = measure_flux(data, (x, y), gain, aperture_radius, r_in, r_out)
                fluxes.append(f)
            if any(f <= 0 for f in fluxes):
                continue
            mags = [-2.5 * np.log10(f) for f in fluxes]
            for i in range(n):
                others_mean = np.mean([mags[j] for j in range(n) if j != i])
                mag_series[i].append(mags[i] - others_mean)
        except Exception:
            continue

    stds = [np.std(s) if len(s) >= 3 else np.inf for s in mag_series]
    print('[비교성 검증] 나머지 비교성 평균 대비 상대등급 std:')
    for i, s in enumerate(stds):
        flag = '  -> 제외' if s > max_std else ''
        print(f'  비교성 #{i}: std={s:.4f}{flag}')

    kept = [xy for xy, s in zip(comp_xy_list, stds) if s <= max_std]
    if len(kept) < 2:
        print('[비교성 검증] 기준 적용 시 2개 미만으로 남아 원래 목록을 유지합니다.')
        return comp_xy_list
    return kept


def optimize_aperture_radius(files, comp_xy_list, radii, gain, annulus_gap=4, annulus_width=6):
    if len(comp_xy_list) < 2:
        raise ValueError('비교성이 2개 이상 있어야 조리개 최적화를 할 수 있습니다.')
    c1_xy, c2_xy = comp_xy_list[0], comp_xy_list[1]
    stds = []
    for r in radii:
        r_in, r_out = r + annulus_gap, r + annulus_gap + annulus_width
        diffs = []
        for path in files:
            try:
                data, _ = load_fits_data(path)
                x1, y1 = refine_centroid(data, *c1_xy, box_size=CENTROID_SEARCH_BOX)
                x2, y2 = refine_centroid(data, *c2_xy, box_size=CENTROID_SEARCH_BOX)
                f1, _, _ = measure_flux(data, (x1, y1), gain, r, r_in, r_out)
                f2, _, _ = measure_flux(data, (x2, y2), gain, r, r_in, r_out)
                if f1 > 0 and f2 > 0:
                    diffs.append((-2.5 * np.log10(f1)) - (-2.5 * np.log10(f2)))
            except Exception:
                continue
        stds.append(np.std(diffs) if len(diffs) >= 3 else np.nan)
    return stds


def process_filter(filter_name, files, target_xy, comp_xy_list, gain, aperture_radius, r_in, r_out):
    rows = []
    for path in files:
        try:
            data, header = load_fits_data(path)
            t_obs = get_obs_time(header)

            tx, ty = refine_centroid(data, *target_xy, box_size=CENTROID_SEARCH_BOX)
            target_shift_px = float(np.hypot(tx - target_xy[0], ty - target_xy[1]))
            t_flux, t_err, _ = measure_flux(data, (tx, ty), gain, aperture_radius, r_in, r_out)

            comp_mags, comp_mag_errs, comp_shifts = [], [], []
            for cx0, cy0 in comp_xy_list:
                cx, cy = refine_centroid(data, cx0, cy0, box_size=CENTROID_SEARCH_BOX)
                comp_shifts.append(float(np.hypot(cx - cx0, cy - cy0)))
                c_flux, c_err, _ = measure_flux(data, (cx, cy), gain, aperture_radius, r_in, r_out)
                c_mag, c_mag_err = instrumental_mag(c_flux, c_err)
                if not np.isnan(c_mag) and c_mag_err > 0:
                    comp_mags.append(c_mag)
                    comp_mag_errs.append(c_mag_err)
            comp_shift_px_max = max(comp_shifts) if comp_shifts else np.nan

            if not comp_mags:
                raise ValueError('모든 비교성 측광에 실패했습니다.')

            weights = 1.0 / np.array(comp_mag_errs)**2
            c_mag = float(np.sum(np.array(comp_mags) * weights) / np.sum(weights))
            c_mag_err = float(np.sqrt(1.0 / np.sum(weights)))

            t_mag, t_mag_err = instrumental_mag(t_flux, t_err)
            diff_mag = t_mag - c_mag
            diff_mag_err = np.sqrt(t_mag_err**2 + c_mag_err**2)

            rows.append({
                'filter': filter_name,
                'file': os.path.basename(path),
                'jd': t_obs.jd,
                'datetime': t_obs.isot,
                'target_flux': t_flux,
                'target_instmag': t_mag,
                'diff_mag': diff_mag,
                'diff_mag_err': diff_mag_err,
                'target_shift_px': target_shift_px,
                'comp_shift_px_max': comp_shift_px_max,
            })
        except Exception as e:
            print(f"[경고] {path} 처리 실패: {e}")
    return pd.DataFrame(rows).sort_values('jd').reset_index(drop=True)


def merge_by_time(df_a, df_b, label_a, label_b, tol_min=TOLERANCE_MIN):
    a = df_a[['jd', 'diff_mag']].rename(columns={'diff_mag': f'mag_{label_a}'}).sort_values('jd')
    b = df_b[['jd', 'diff_mag']].rename(columns={'diff_mag': f'mag_{label_b}'}).sort_values('jd')
    return pd.merge_asof(a, b, on='jd', direction='nearest', tolerance=tol_min / (24 * 60)).dropna()


# ============================================================
# 실행 흐름
# ============================================================

def main():
    global GAIN_DEFAULT, APERTURE_RADIUS, ANNULUS_R_IN, ANNULUS_R_OUT

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- (옵션) gain 실측 ---
    if os.path.exists(FLAT_FILE_1) and os.path.exists(FLAT_FILE_2):
        try:
            g, m1, m2 = measure_gain(FLAT_FILE_1, FLAT_FILE_2)
            print(f'[gain] 프레임1 평균={m1:.1f} ADU, 프레임2 평균={m2:.1f} ADU')
            print(f'[gain] 측정된 gain={g:.4f} e-/ADU 로 갱신')
            GAIN_DEFAULT = g
        except Exception as e:
            print(f'[gain] 측정 실패({e}), 기존 값 GAIN_DEFAULT={GAIN_DEFAULT} 사용')
    else:
        print(f'[gain] flat 파일을 찾지 못해 건너뜁니다 (GAIN_DEFAULT={GAIN_DEFAULT} 그대로 사용).')

    # --- 5. 미리보기 & 별 자동 탐지 ---
    preview_file = list_fits_files(BASE_DIR, FILE_PATTERN_BY_FILTER[PREVIEW_FILTER])[0]
    data, header = load_fits_data(preview_file)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    print(f'[미리보기] 이미지: {preview_file}')
    print(f'[미리보기] 배경 median={median:.2f}, std={std:.2f}')

    finder = DAOStarFinder(fwhm=DETECT_FWHM, threshold=DETECT_THRESHOLD_SIGMA * std)
    sources = finder(data - median)
    if sources is None:
        print('별을 하나도 찾지 못했습니다. DETECT_THRESHOLD_SIGMA를 낮추거나 DETECT_FWHM을 조정해서 다시 실행하세요.')
        sys.exit(1)

    x_col = 'xcentroid' if 'xcentroid' in sources.colnames else 'x_centroid'
    y_col = 'ycentroid' if 'ycentroid' in sources.colnames else 'y_centroid'
    sources.sort('flux')
    sources.reverse()

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(data)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    for i, s in enumerate(sources[:40]):
        ax.plot(s[x_col], s[y_col], 'o', mfc='none', mec='red', ms=12)
        ax.text(s[x_col] + 6, s[y_col] + 6, str(i), color='yellow', fontsize=9)
    ax.set_title(f'{PREVIEW_FILTER} filter, first frame — detected stars (brightness rank)')
    preview_png = os.path.join(OUTPUT_DIR, 'preview_stars.png')
    fig.savefig(preview_png, dpi=120)
    plt.close(fig)

    print(f'[미리보기] 별 탐지 이미지 저장: {preview_png}')
    print(f'[미리보기] 탐지된 별 개수: {len(sources)}')
    print(sources[['id', x_col, y_col, 'flux']][:40])

    if TARGET_ID is None or COMP_IDS is None:
        print()
        print('=' * 70)
        print(f'{preview_png} 파일을 열어서 대상 별과 비교성 번호를 확인하세요.')
        print('그 다음 이 스크립트 상단의 TARGET_ID / COMP_IDS 값을 채우고 다시 실행하세요.')
        print('=' * 70)
        return

    target_xy = (float(sources[TARGET_ID][x_col]), float(sources[TARGET_ID][y_col]))
    comp_xy_list = [(float(sources[i][x_col]), float(sources[i][y_col])) for i in COMP_IDS]
    print('대상 별 좌표:', target_xy)
    print('비교성 좌표들:', comp_xy_list)

    # --- (옵션) 비교성 안정성 검증 ---
    vet_files = list_fits_files(BASE_DIR, FILE_PATTERN_BY_FILTER[PREVIEW_FILTER])[:VET_SAMPLE_SIZE]
    comp_xy_list = vet_comparison_stars(vet_files, comp_xy_list, GAIN_DEFAULT,
                                         APERTURE_RADIUS, ANNULUS_R_IN, ANNULUS_R_OUT)
    print('최종 비교성 좌표들:', comp_xy_list)

    # --- (옵션) 조리개 반지름 자동 최적화 ---
    aperture_files = list_fits_files(BASE_DIR, FILE_PATTERN_BY_FILTER[PREVIEW_FILTER])[:APERTURE_SAMPLE_SIZE]
    radii = list(range(6, 23, 2))
    stds = optimize_aperture_radius(aperture_files, comp_xy_list, radii, GAIN_DEFAULT)
    valid = [(r, s) for r, s in zip(radii, stds) if not np.isnan(s)]
    if valid:
        best_radius, best_std = min(valid, key=lambda t: t[1])
        APERTURE_RADIUS = best_radius
        ANNULUS_R_IN = best_radius + 4
        ANNULUS_R_OUT = best_radius + 10
        print(f'[조리개 최적화] APERTURE_RADIUS={APERTURE_RADIUS}, ANNULUS_R_IN={ANNULUS_R_IN}, '
              f'ANNULUS_R_OUT={ANNULUS_R_OUT} (std={best_std:.4f})')

        fig, ax = plt.subplots(figsize=(7, 4))
        rs, ss = zip(*valid)
        ax.plot(rs, ss, 'o-')
        ax.axvline(best_radius, color='r', ls='--', label=f'최적 반지름 = {best_radius}px')
        ax.set_xlabel('Aperture radius (px)')
        ax.set_ylabel('Comp1-Comp2 magnitude scatter (std)')
        ax.set_title('Aperture Radius Optimization')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'aperture_optimization.png'), dpi=120)
        plt.close(fig)
    else:
        print('[조리개 최적화] 실패 (측광 가능한 프레임 부족). 기존 값을 유지합니다.')

    # --- 6. 필터별 시계열 측광 실행 ---
    results = {}
    for fname in FILTER_ORDER:
        files = list_fits_files(BASE_DIR, FILE_PATTERN_BY_FILTER[fname])
        print(f'--- {fname} 필터 처리 중 ({len(files)}장) ---')
        results[fname] = process_filter(fname, files, target_xy, comp_xy_list,
                                         GAIN_DEFAULT, APERTURE_RADIUS, ANNULUS_R_IN, ANNULUS_R_OUT)
        print(f'{fname}: {len(results[fname])}장 처리 완료')

    all_df = pd.concat(results.values(), ignore_index=True)

    # --- 6.5 아웃라이어 프레임 자동 제거 ---
    for fname in FILTER_ORDER:
        df = results.get(fname)
        if df is None or df.empty:
            continue
        median_err = df['diff_mag_err'].median()
        err_threshold = max(median_err * MAX_DIFF_MAG_ERR_MULTIPLIER, MIN_DIFF_MAG_ERR_FLOOR)
        bad = ((df['target_shift_px'] > MAX_CENTROID_SHIFT_PX) |
               (df['comp_shift_px_max'] > MAX_CENTROID_SHIFT_PX) |
               (df['diff_mag_err'] > err_threshold))
        n_bad = int(bad.sum())
        if n_bad > 0:
            print(f'--- {fname} 필터: {n_bad}/{len(df)}장 제외 (오차 임계값 {err_threshold:.4f}mag) ---')
            for _, row in df[bad].iterrows():
                print(f"  {row['file']}: shift={row['target_shift_px']:.1f}/{row['comp_shift_px_max']:.1f}px, "
                      f"err={row['diff_mag_err']:.4f}mag")
        else:
            print(f'--- {fname} 필터: 제외된 프레임 없음 ({len(df)}장 모두 유지) ---')
        results[fname] = df[~bad].reset_index(drop=True)

    all_df = pd.concat(results.values(), ignore_index=True)
    print(f'\n필터링 후 총 {len(all_df)}장')

    # --- 7. 결과 저장 (CSV) ---
    all_df.to_csv(OUTPUT_CSV, index=False)
    print(f'저장 완료: {OUTPUT_CSV}')

    # --- 8. 필터별 광도곡선 그래프 ---
    cmap = plt.cm.tab10
    colors = {f: cmap(i % 10) for i, f in enumerate(FILTER_ORDER)}
    t0 = all_df['jd'].min()

    fig, axes = plt.subplots(len(results), 1, figsize=(10, 4 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
    for ax, fname in zip(axes, FILTER_ORDER):
        df = results.get(fname)
        if df is None or df.empty:
            continue
        hours = (df['jd'] - t0) * 24.0
        ax.errorbar(hours, df['diff_mag'], yerr=df['diff_mag_err'],
                     fmt='o', ms=4, capsize=2, color=colors.get(fname, 'k'), label=f'{fname} filter')
        ax.invert_yaxis()
        ax.set_ylabel('Differential Magnitude')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(f'Time (hours since JD {t0:.5f})')
    fig.suptitle(f"Filter-by-Filter Light Curve (Differential Photometry) — {'/'.join(FILTER_ORDER)}")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'lightcurve_by_filter.png'), dpi=120)
    plt.close(fig)

    # --- 9. 통합 그래프 ---
    fig, ax = plt.subplots(figsize=(11, 6))
    for fname in FILTER_ORDER:
        df = results.get(fname)
        if df is None or df.empty:
            continue
        hours = (df['jd'] - t0) * 24.0
        ax.errorbar(hours, df['diff_mag'], yerr=df['diff_mag_err'],
                     fmt='o-', ms=4, capsize=2, color=colors.get(fname, 'k'),
                     label=f'{fname} filter', alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel(f'Time (hours since JD {t0:.5f})')
    ax.set_ylabel('Differential Magnitude')
    ax.set_title(f"Combined {'/'.join(FILTER_ORDER)} Light Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'lightcurve_combined.png'), dpi=120)
    plt.close(fig)

    # --- 10. 색지수 시계열 ---
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (f1, f2) in enumerate(zip(FILTER_ORDER, FILTER_ORDER[1:])):
        df1, df2 = results.get(f1), results.get(f2)
        if df1 is None or df2 is None or df1.empty or df2.empty:
            continue
        merged = merge_by_time(df1, df2, f1, f2)
        if merged.empty:
            continue
        label = f'{f1}-{f2}'
        merged[label] = merged[f'mag_{f1}'] - merged[f'mag_{f2}']
        ax.plot((merged['jd'] - t0) * 24.0, merged[label], 'o-', label=label, color=cmap((i + 3) % 10))
    ax.set_xlabel(f'Time (hours since JD {t0:.5f})')
    ax.set_ylabel('Color Index')
    ax.set_title('Color Index vs Time')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'color_index.png'), dpi=120)
    plt.close(fig)

    print()
    print('모든 그래프/CSV를', OUTPUT_DIR, '에 저장했습니다:')
    print('  preview_stars.png, lightcurve_by_filter.png, lightcurve_combined.png,')
    print('  color_index.png, photometry_results.csv')


if __name__ == '__main__':
    main()
