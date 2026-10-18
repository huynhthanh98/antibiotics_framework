"""
Parse training log file HOẶC run_metrics.csv → trích xuất Image-Level summary → lưu Excel.

Usage:
    # Từ log file (cần có dòng "TỔNG HỢP 5 SEEDS" và "[Image-Level (Vote)]"):
    python parse_log.py training_log_CIP.txt --out summary.xlsx --both

    # Từ thư mục results (khi log chưa đủ hoặc không parse được):
    python parse_log.py --from-results /home/son/new_mia/results --task 1 --drug CIP --out summary.xlsx --both
"""

import re
import sys
import argparse
import csv
from pathlib import Path

import openpyxl
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.utils import get_column_letter


METRICS_ORDER = [
    ('Recall (Resistant)',    'recall_resistant'),
    ('Precision (Resistant)', 'precision_resistant'),
    ('F1 (Resistant)',        'f1_resistant'),
    ('Recall (Sensitive)',    'recall_sensitive'),
    ("Youden's J",            'youden_j'),
    ('F1 Macro',              'f1_macro'),
    ('AUC',                   'auc'),
    ('Accuracy',              'accuracy'),
    ('MCC',                   'mcc'),
]

# Pattern khớp: "  Recall    (Resistant)    : 0.9857 ± 0.0286" hoặc "    Recall..."
METRIC_RE = re.compile(r"^\s*(.+?)\s*:\s*([\d.]+)\s*[±+]\s*([\d.]+)")

# Pattern khớp: "  TỔNG HỢP 5 SEEDS | Task1 | CIP | resnet50"
HEADER_RE = re.compile(r"TỔNG\s*HỢP\s+\d+\s+SEEDS\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)", re.IGNORECASE)


def parse_from_results(results_dir: str, task: int = 1, drug: str = "CIP") -> tuple[list[dict], list[dict]]:
    """
    Đọc run_metrics.csv từ results/task{N}/{drug}/{model}/ → tính mean ± std.

    Returns:
        (results_image, results_cell)
    """
    try:
        import pandas as pd
    except ImportError:
        print("  [Warning] Cần cài pandas: pip install pandas")
        return [], []

    results_dir = Path(results_dir)
    task_dir = results_dir / f"task{task}" / drug
    if not task_dir.exists():
        print(f"  [Warning] Không tìm thấy: {task_dir}")
        return [], []

    all_rows = []
    for model_dir in sorted(task_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_path = model_dir / "run_metrics.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        df['model'] = model_dir.name
        all_rows.append(df)

    if not all_rows:
        return [], []

    full = pd.concat(all_rows, ignore_index=True)

    metric_cols = ['recall_resistant', 'precision_resistant', 'f1_resistant', 'recall_sensitive',
                   'youden_j', 'f1_macro', 'f1_weighted', 'auc', 'accuracy', 'mcc']

    def agg_level(level_val: str) -> list[dict]:
        sub = full[full['level'] == level_val]
        if sub.empty:
            return []
        out = []
        for model in sub['model'].unique():
            msub = sub[sub['model'] == model]
            rec = {
                'task': f'Task{task}',
                'drug': drug,
                'model': model,
            }
            for col in metric_cols:
                if col not in msub.columns:
                    continue
                vals = msub[col].dropna()
                if len(vals) > 0:
                    rec[f'{col}_mean'] = float(vals.mean())
                    rec[f'{col}_std'] = float(vals.std()) if len(vals) > 1 else 0.0
                    rec[col] = f"{rec[f'{col}_mean']:.4f} ± {rec[f'{col}_std']:.4f}"
            out.append(rec)
        return out

    results_img = agg_level('image_level')
    results_cell = agg_level('cell_level')
    return results_img, results_cell


def parse_log(log_path: str, level: str = 'image') -> list[dict]:
    """
    Đọc log file, trích xuất kết quả Image-Level (hoặc Cell-Level) tổng hợp.

    Args:
        log_path : Đường dẫn file log
        level    : 'image' → [Image-Level (Vote)] | 'cell' → [Cell-Level]

    Returns:
        List[dict] mỗi dict là kết quả một model
    """
    target_section = '[Image-Level (Vote)]' if level == 'image' else '[Cell-Level]'
    results = []

    with open(log_path, encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')

        # Tìm header TỔNG HỢP
        m_header = HEADER_RE.search(line)
        if m_header:
            task      = m_header.group(1).strip()
            drug      = m_header.group(2).strip()
            model     = m_header.group(3).strip()

            # Tìm section [Image-Level (Vote)] hoặc [Cell-Level] trong 20 dòng tiếp
            found_section = False
            for j in range(i + 1, min(i + 25, len(lines))):
                if target_section in lines[j]:
                    found_section = True
                    # Parse metrics từ đây
                    record = {'task': task, 'drug': drug, 'model': model}
                    for k in range(j + 1, min(j + 15, len(lines))):
                        m_metric = METRIC_RE.match(lines[k])
                        if m_metric:
                            raw_name = m_metric.group(1).strip()
                            mean_val = float(m_metric.group(2))
                            std_val  = float(m_metric.group(3))
                            # Map tên hiển thị → key nội bộ
                            key = _map_metric_name(raw_name)
                            record[f'{key}_mean'] = mean_val
                            record[f'{key}_std']  = std_val
                            record[key]           = f"{mean_val:.4f} ± {std_val:.4f}"
                        elif lines[k].strip().startswith('[') or lines[k].strip() == '':
                            break
                    results.append(record)
                    break

            if not found_section:
                print(f"  [Warning] Không tìm thấy '{target_section}' cho {task}|{drug}|{model}")
        i += 1

    return results


def _map_metric_name(raw: str) -> str:
    """Map tên hiển thị trong log → key nội bộ."""
    raw = raw.strip().lower()
    raw = re.sub(r'\s+', ' ', raw)
    mappings = {
        'recall (resistant)':    'recall_resistant',
        'precision (resistant)': 'precision_resistant',
        'f1 (resistant)':        'f1_resistant',
        'recall (sensitive)':    'recall_sensitive',
        "youden's j":            'youden_j',
        'f1 macro':              'f1_macro',
        'f1 weighted':           'f1_weighted',
        'auc':                   'auc',
        'accuracy':              'accuracy',
        'mcc':                   'mcc',
    }
    # Khớp linh hoạt (xử lý khoảng trắng thừa trong log)
    for pattern, key in mappings.items():
        if pattern in raw:
            return key
    return raw.replace(' ', '_')


def save_csv(results: list[dict], out_path: str):
    """Lưu kết quả ra CSV. Mỗi metric = 1 cột dạng "mean ± std"."""
    if not results:
        print("Không có kết quả để lưu.")
        return

    metric_keys = [k for _, k in METRICS_ORDER if any(k in r for r in results)]
    fieldnames  = ['task', 'drug', 'model'] + metric_keys

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ Đã lưu CSV  → {out_path}")


# ─── Excel styles ─────────────────────────────────────────────────────────────

_HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")   # xanh đậm
_GROUP_FILLS  = [
    PatternFill("solid", fgColor="D6E4F0"),  # nhóm chẵn: xanh nhạt
    PatternFill("solid", fgColor="FFFFFF"),  # nhóm lẻ : trắng
]
_BEST_FILL    = PatternFill("solid", fgColor="E2EFDA")   # xanh lá nhạt (highlight best)
_HEADER_FONT  = Font(name="Calibri", bold=True, color="FFFFFF", size=10)
_BODY_FONT    = Font(name="Calibri", size=10)
_CENTER       = Alignment(horizontal="center", vertical="center", wrap_text=False)
_LEFT         = Alignment(horizontal="left",   vertical="center")
_BORDER_SIDE  = Side(style="thin", color="BFBFBF")
_THIN_BORDER  = Border(
    left=_BORDER_SIDE, right=_BORDER_SIDE,
    top=_BORDER_SIDE,  bottom=_BORDER_SIDE,
)


def _write_sheet(ws, results: list[dict], level_label: str):
    """Ghi một sheet kết quả vào workbook. Mỗi metric = 1 ô dạng "mean ± std"."""
    active_keys = [k for _, k in METRICS_ORDER
                   if any(f'{k}_mean' in r for r in results)]
    active_displays = {k: d for d, k in METRICS_ORDER}

    # ── Tiêu đề sheet ─────────────────────────────────────────────────────────
    ws.append([f"Kết quả {level_label} — trích xuất từ training log"])
    ws.merge_cells(start_row=1, start_column=1,
                   end_row=1, end_column=3 + len(active_keys))
    title_cell = ws.cell(1, 1)
    title_cell.font      = Font(name="Calibri", bold=True, size=12, color="1F4E79")
    title_cell.alignment = _CENTER
    ws.row_dimensions[1].height = 20

    # ── Header row: 1 cột per metric ──────────────────────────────────────────
    header_row = ['Task', 'Drug', 'Model']
    for k in active_keys:
        header_row.append(active_displays.get(k, k))
    ws.append(header_row)

    hrow = ws.max_row
    for col_idx, val in enumerate(header_row, start=1):
        cell = ws.cell(hrow, col_idx, val)
        cell.font      = _HEADER_FONT
        cell.fill      = _HEADER_FILL
        cell.alignment = _CENTER
        cell.border    = _THIN_BORDER
    ws.row_dimensions[hrow].height = 30

    # ── Data rows: mỗi ô = "mean ± std" ───────────────────────────────────────
    best_row_idx = None
    if 'recall_resistant' in active_keys:
        recall_idx = active_keys.index('recall_resistant')
        best_val = -1.0
        for row_idx, r in enumerate(results):
            m = r.get(f'recall_resistant_mean')
            if m is not None and m > best_val:
                best_val, best_row_idx = m, row_idx

    for row_idx, r in enumerate(results):
        data_row = [r.get('task', ''), r.get('drug', ''), r.get('model', '')]
        for k in active_keys:
            # Dùng chuỗi "mean ± std" (đã có sẵn trong record[key])
            data_row.append(r.get(k, ''))
        ws.append(data_row)

        drow = ws.max_row
        fill = _BEST_FILL if row_idx == best_row_idx else _GROUP_FILLS[row_idx % 2]
        for col_idx, val in enumerate(data_row, start=1):
            cell = ws.cell(drow, col_idx, val)
            cell.font      = _BODY_FONT
            cell.fill      = fill
            cell.border    = _THIN_BORDER
            if col_idx <= 3:
                cell.alignment = _LEFT
            else:
                cell.alignment = _CENTER

    # ── Column widths ─────────────────────────────────────────────────────────
    col_widths = [10, 8, 22] + [18] * len(active_keys)
    for i, width in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width

    # Freeze header
    ws.freeze_panes = ws.cell(hrow + 1, 1)


def save_excel(results_image: list[dict], results_cell: list[dict],
               out_path: str, both: bool = False):
    """Lưu kết quả ra file Excel với định dạng đẹp."""
    wb = openpyxl.Workbook()

    if both and results_cell:
        # Sheet 1: Image-Level
        ws_img = wb.active
        ws_img.title = "Image-Level (Vote)"
        _write_sheet(ws_img, results_image, "Image-Level (Vote)")
        # Sheet 2: Cell-Level
        ws_cell = wb.create_sheet("Cell-Level")
        _write_sheet(ws_cell, results_cell, "Cell-Level")
    else:
        ws = wb.active
        if results_image:
            ws.title = "Image-Level (Vote)"
            _write_sheet(ws, results_image, "Image-Level (Vote)")
        else:
            ws.title = "Cell-Level"
            _write_sheet(ws, results_cell, "Cell-Level")

    wb.save(out_path)
    print(f"✓ Đã lưu Excel → {out_path}")


def save_excel_compare_tasks(
    results_img_t1: list[dict],
    results_img_t2: list[dict],
    results_cell_t1: list[dict],
    results_cell_t2: list[dict],
    out_path: str,
):
    """
    Lưu file Excel so sánh Task1 vs Task2 cho cùng một drug.

    Layout:
        - Mỗi backbone = 2 dòng: Task1 (trên), Task2 (dưới)
        - Với mỗi metric, ô nào có mean tốt hơn sẽ được highlight.
    """
    wb = openpyxl.Workbook()

    def _write_compare_sheet(ws, res_t1: list[dict], res_t2: list[dict], level_label: str):
        if not res_t1 and not res_t2:
            ws.append([f"Không có kết quả {level_label}."])
            return

        # Map model → record cho từng task
        by_model_t1 = {r['model']: r for r in res_t1}
        by_model_t2 = {r['model']: r for r in res_t2}
        all_models = sorted(set(by_model_t1.keys()) | set(by_model_t2.keys()))

        # Các metric đang có
        active_keys = []
        for _, k in METRICS_ORDER:
            if any(f'{k}_mean' in r for r in res_t1) or any(f'{k}_mean' in r for r in res_t2):
                active_keys.append(k)
        active_displays = {k: d for d, k in METRICS_ORDER}

        # Tiêu đề sheet
        ws.append([f"So sánh Task1 vs Task2 — {level_label}"])
        ws.merge_cells(start_row=1, start_column=1,
                       end_row=1, end_column=3 + len(active_keys))
        title_cell = ws.cell(1, 1)
        title_cell.font = Font(name="Calibri", bold=True, size=12, color="1F4E79")
        title_cell.alignment = _CENTER
        ws.row_dimensions[1].height = 20

        # Header
        header_row = ['Task', 'Drug', 'Model']
        for k in active_keys:
            header_row.append(active_displays.get(k, k))
        ws.append(header_row)

        hrow = ws.max_row
        for col_idx, val in enumerate(header_row, start=1):
            cell = ws.cell(hrow, col_idx, val)
            cell.font = _HEADER_FONT
            cell.fill = _HEADER_FILL
            cell.alignment = _CENTER
            cell.border = _THIN_BORDER
        ws.row_dimensions[hrow].height = 30

        # Data: mỗi backbone = 2 dòng: Task1, Task2
        for grp_idx, model in enumerate(all_models):
            r1 = by_model_t1.get(model)
            r2 = by_model_t2.get(model)

            rows = []
            if r1:
                rows.append(('Task1', r1))
            if r2:
                rows.append(('Task2', r2))

            row_indices = []
            for row_task, rec in rows:
                data_row = [row_task, rec.get('drug', ''), model]
                for k in active_keys:
                    data_row.append(rec.get(k, ''))
                ws.append(data_row)

                drow = ws.max_row
                row_indices.append(drow)
                base_fill = _GROUP_FILLS[grp_idx % 2]
                for col_idx, val in enumerate(data_row, start=1):
                    cell = ws.cell(drow, col_idx, val)
                    cell.font = _BODY_FONT
                    cell.fill = base_fill
                    cell.border = _THIN_BORDER
                    if col_idx <= 3:
                        cell.alignment = _LEFT
                    else:
                        cell.alignment = _CENTER

            # Highlight metric tốt hơn trong từng cặp Task1/Task2
            if len(row_indices) == 2:
                row_t1, row_t2 = row_indices
                rec1 = by_model_t1.get(model)
                rec2 = by_model_t2.get(model)
                if rec1 and rec2:
                    for idx_k, k in enumerate(active_keys, start=1):
                        m1 = rec1.get(f'{k}_mean')
                        m2 = rec2.get(f'{k}_mean')
                        col = 3 + idx_k
                        if m1 is None and m2 is None:
                            continue
                        if m2 is None or (m1 is not None and m1 > m2):
                            best_row = row_t1
                        elif m1 is None or m2 > m1:
                            best_row = row_t2
                        else:
                            best_row = None
                        if best_row is not None:
                            best_cell = ws.cell(best_row, col)
                            best_cell.fill = _BEST_FILL

        # Column widths + freeze header
        col_widths = [8, 8, 22] + [18] * len(active_keys)
        for i, width in enumerate(col_widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = width
        ws.freeze_panes = ws.cell(hrow + 1, 1)

    # Sheet 1: Image-Level
    ws_img = wb.active
    ws_img.title = "Image-Level T1 vs T2"
    _write_compare_sheet(ws_img, results_img_t1, results_img_t2, "Image-Level (Vote)")

    # Sheet 2: Cell-Level (nếu có)
    if results_cell_t1 or results_cell_t2:
        ws_cell = wb.create_sheet("Cell-Level T1 vs T2")
        _write_compare_sheet(ws_cell, results_cell_t1, results_cell_t2, "Cell-Level")

    wb.save(out_path)
    print(f"✓ Đã lưu Excel (Task1 vs Task2) → {out_path}")


def print_table(results: list[dict], level: str):
    """In bảng đẹp ra terminal."""
    if not results:
        print("Không có kết quả.")
        return

    level_label = 'Image-Level (Vote)' if level == 'image' else 'Cell-Level'
    print(f"\n{'═'*120}")
    print(f"  KẾT QUẢ {level_label.upper()} - TỔNG HỢP THEO MODEL")
    print(f"{'═'*120}")

    # Header
    header = f"  {'Model':<22} {'Task':<8} {'Drug':<6}"
    for display, key in METRICS_ORDER:
        if any(f'{key}_mean' in r for r in results):
            header += f"  {display:<22}"
    print(header)
    print(f"  {'-'*116}")

    for r in results:
        row = f"  {r['model']:<22} {r['task']:<8} {r['drug']:<6}"
        for display, key in METRICS_ORDER:
            val = r.get(key, 'N/A')
            if val != 'N/A':
                row += f"  {val:<22}"
        print(row)

    print(f"{'═'*120}\n")


def main():
    parser = argparse.ArgumentParser(description='Parse training log hoặc run_metrics.csv → Excel')
    parser.add_argument('log_file', nargs='?', default=None,
                        help='Đường dẫn file log (e.g. training_log_CIP.txt)')
    parser.add_argument('--from-results', type=str, default=None,
                        help='Lấy từ run_metrics.csv trong thư mục results (khi log không parse được)')
    parser.add_argument('--task', type=int, default=1, help='Task (1 hoặc 2). Dùng với --from-results')
    parser.add_argument('--drug', type=str, default='CIP', help='Drug (CIP, TE, AZI). Dùng với --from-results')
    parser.add_argument('--out',   default=None,
                        help='Tên file output. Default: summary.xlsx')
    parser.add_argument('--level', choices=['image', 'cell'], default='image',
                        help='image = Image-Level | cell = Cell-Level. Default: image')
    parser.add_argument('--both', action='store_true',
                        help='Lưu cả Image-Level và Cell-Level vào 2 sheet')
    parser.add_argument(
        '--compare-tasks',
        action='store_true',
        help='So sánh Task1 vs Task2 cho cùng 1 drug (chỉ dùng với --from-results)',
    )
    args = parser.parse_args()

    use_xlsx = True
    results_img, results_cell = [], []

    # ── Mode: so sánh Task1 vs Task2 cho cùng 1 drug ────────────────────────────
    if args.compare_tasks:
        if not args.from_results:
            print("Lỗi: --compare-tasks cần kèm --from-results /path/to/results")
            sys.exit(1)
        results_dir = Path(args.from_results)
        if not results_dir.exists():
            print(f"Lỗi: không tìm thấy thư mục '{results_dir}'")
            sys.exit(1)
        print(f"\nSo sánh Task1 vs Task2 từ run_metrics.csv: {results_dir} (drug={args.drug})")
        img_t1, cell_t1 = parse_from_results(str(results_dir), task=1, drug=args.drug)
        img_t2, cell_t2 = parse_from_results(str(results_dir), task=2, drug=args.drug)
        print(f"Tìm thấy {len(img_t1)} model(s) cho Task1, {len(img_t2)} model(s) cho Task2.")
        if not img_t1 and not img_t2 and not cell_t1 and not cell_t2:
            print("Không có kết quả để so sánh.")
            sys.exit(1)
        out_path = args.out or f"results_task1_vs_task2_{args.drug}_summary.xlsx"
        save_excel_compare_tasks(img_t1, img_t2, cell_t1, cell_t2, out_path)
        sys.exit(0)

    # ── Mode bình thường ────────────────────────────────────────────────────────
    if args.from_results:
        # Parse từ run_metrics.csv
        results_dir = Path(args.from_results)
        if not results_dir.exists():
            print(f"Lỗi: không tìm thấy thư mục '{results_dir}'")
            sys.exit(1)
        print(f"\nParsing từ run_metrics.csv: {results_dir} (task={args.task}, drug={args.drug})")
        results_img, results_cell = parse_from_results(str(results_dir), task=args.task, drug=args.drug)
        print(f"Tìm thấy {len(results_img)} model(s).")
        out_path = args.out or f"results_task{args.task}_{args.drug}_summary.xlsx"
    elif args.log_file:
        log_path = args.log_file
        if not Path(log_path).exists():
            print(f"Lỗi: không tìm thấy file '{log_path}'")
            sys.exit(1)
        stem     = str(Path(log_path).with_suffix(''))
        out_path = args.out or f"{stem}_summary.xlsx"
        use_xlsx = out_path.endswith('.xlsx')
        print(f"\nParsing log: {log_path}")

        if args.both:
            results_img  = parse_log(log_path, level='image')
            results_cell = parse_log(log_path, level='cell')
            print(f"Tìm thấy {len(results_img)} model(s).")
            if not results_img and not results_cell:
                print("  [Gợi ý] Log có thể chưa đủ (chưa có TỔNG HỢP 5 SEEDS). Thử: --from-results /path/to/results")
        else:
            results_img = parse_log(log_path, level=args.level)
            print(f"Tìm thấy {len(results_img)} model(s).")
            if not results_img:
                print("  [Gợi ý] Thử: --from-results /path/to/results")
    else:
        print("Lỗi: Cần chỉ định log_file hoặc --from-results")
        parser.print_help()
        sys.exit(1)

    if not results_img and not results_cell:
        print("Không có kết quả để lưu.")
        sys.exit(1)

    print_table(results_img,  'image')
    print_table(results_cell, 'cell')
    if use_xlsx:
        save_excel(results_img, results_cell, out_path, both=(args.both or bool(results_cell)))
    else:
        stem = str(Path(out_path).with_suffix(''))
        if results_img:
            save_csv(results_img, f"{stem}_image_level.csv")
        if results_cell:
            save_csv(results_cell, f"{stem}_cell_level.csv")


if __name__ == '__main__':
    main()
