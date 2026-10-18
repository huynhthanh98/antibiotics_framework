#!/bin/bash
# ============================================================
#  Run ALL models × ALL drugs - Task 1 với Erosion Augmentation
#
#  Giống run_all_models.sh nhưng thêm --mask_erosion_aug để
#  tăng robustness khi test set có mask không hoàn hảo.
#
#  Kết quả lưu vào: results_erosion_aug/ (tách biệt với results/)
#  để so sánh trực tiếp:
#    results/task1/         → baseline (không erosion aug)
#    results_erosion_aug/task1/ → sau khi erosion aug
#
#  Drugs  : AZI, CIP, TE
#  Params : seeds 0-4 | threshold pr_equal | best_metric f1_resistant
#           best_metric_level image_level   | loss focal | BF16
#           mask_erosion_aug ON | max_px 3
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="${SCRIPT_DIR}/antibiotics_framework"
LOG_DIR="${SCRIPT_DIR}/logs_task1_erosion_aug"
OUTPUT_DIR="${SCRIPT_DIR}/results_erosion_aug"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ─── Tham số cố định ──────────────────────────────────────
TASK=1
SEEDS="0 1 2 3 4"
THRESHOLD_METHOD="pr_equal"
BEST_METRIC="f1_resistant"
BEST_METRIC_LEVEL="image_level"
LOSS="focal"

# ─── BF16 mode ────────────────────────────────────────────
USE_BF16=true

# ─── Erosion augmentation ─────────────────────────────────
EROSION_MAX_PX=3       # Erode ngẫu nhiên 1–3px, xác suất 50% mỗi sample train

# ─── Danh sách drugs ──────────────────────────────────────
DRUGS=("CIP" "AZI" "TE")
DRUGS=("TE")
# ─── Log files ────────────────────────────────────────────
PROGRESS_LOG="${LOG_DIR}/progress.log"
STATUS_FILE="${LOG_DIR}/status.txt"

# ─── Hàm ghi log có timestamp ─────────────────────────────
log() {
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "${ts} | ${msg}" | tee -a "${PROGRESS_LOG}"
}

log_drug() {
    local msg="$*"
    echo "" | tee -a "${PROGRESS_LOG}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${PROGRESS_LOG}"
    log "  [DRUG] ${msg}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${PROGRESS_LOG}"
}

# ─── Batch size cho từng model (giống run_all_models.sh) ──
get_batch_size() {
    local model=$1
    if [ "${USE_BF16}" = "true" ]; then
        case "${model}" in
            resnet50)           echo 96  ;;
            resnet101)          echo 64  ;;
            vgg16)              echo 48  ;;
            vgg19)              echo 48  ;;
            convnext_tiny)      echo 96  ;;
            convnext_small)     echo 64  ;;
            convnext_base)      echo 48  ;;
            convnextv2_tiny)    echo 96  ;;
            convnextv2_small)   echo 64  ;;
            convnextv2_base)    echo 48  ;;
            densenet121)        echo 96  ;;
            densenet169)        echo 64  ;;
            densenet201)        echo 48  ;;
            efficientnet_b1)    echo 128 ;;
            efficientnet_b3)    echo 64  ;;
            efficientnet_b5)    echo 32  ;;
            efficientnet_b7)    echo 24  ;;
            swin_tiny)          echo 96  ;;
            swin_small)         echo 64  ;;
            swin_base)          echo 48  ;;
            vit_small)          echo 96  ;;
            vit_base)           echo 48  ;;
            *)                  echo 48  ;;
        esac
    else
        case "${model}" in
            resnet50)           echo 64  ;;
            resnet101)          echo 48  ;;
            vgg16)              echo 32  ;;
            vgg19)              echo 32  ;;
            convnext_tiny)      echo 64  ;;
            convnext_small)     echo 48  ;;
            convnext_base)      echo 32  ;;
            convnextv2_tiny)    echo 64  ;;
            convnextv2_small)   echo 48  ;;
            convnextv2_base)    echo 32  ;;
            densenet121)        echo 64  ;;
            densenet169)        echo 48  ;;
            densenet201)        echo 32  ;;
            efficientnet_b1)    echo 96  ;;
            efficientnet_b3)    echo 48  ;;
            efficientnet_b5)    echo 24  ;;
            efficientnet_b7)    echo 16  ;;
            swin_tiny)          echo 64  ;;
            swin_small)         echo 48  ;;
            swin_base)          echo 32  ;;
            vit_small)          echo 64  ;;
            vit_base)           echo 32  ;;
            *)                  echo 32  ;;
        esac
    fi
}

# ─── Danh sách TẤT CẢ model ───────────────────────────────
ALL_MODELS=(
    resnet50 resnet101
    vgg16 vgg19
    convnext_tiny convnext_small convnext_base
    convnextv2_tiny convnextv2_small convnextv2_base
    densenet121 densenet169 densenet201
    efficientnet_b1 efficientnet_b3 efficientnet_b5 efficientnet_b7
    swin_tiny swin_small swin_base
    vit_small vit_base
)

BF16_FLAG=""
[ "${USE_BF16}" = "true" ] && BF16_FLAG="--bf16"

TOTAL_MODELS=${#ALL_MODELS[@]}
TOTAL_DRUGS=${#DRUGS[@]}
TOTAL_RUNS=$(( TOTAL_DRUGS * TOTAL_MODELS ))
RUN_IDX=0
FAILED_RUNS=()

declare -A RUN_STATUS
declare -A RUN_DURATION

SCRIPT_START=$(date +%s)

# ─── Khởi tạo progress log ────────────────────────────────
> "${PROGRESS_LOG}"
log "╔══════════════════════════════════════════════════════════╗"
log "║  TASK 1 + EROSION AUGMENTATION — FULL MODEL SWEEP       ║"
log "╚══════════════════════════════════════════════════════════╝"
log "  GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "  Drugs      : ${DRUGS[*]}"
log "  Models     : ${TOTAL_MODELS} × ${TOTAL_DRUGS} drugs = ${TOTAL_RUNS} runs"
log "  Seeds      : ${SEEDS}"
log "  Precision  : $([ "${USE_BF16}" = "true" ] && echo 'BFloat16' || echo 'Float32')"
log "  Loss/Thresh: ${LOSS} / ${THRESHOLD_METHOD}"
log "  ErosionAug : ON | max_px=${EROSION_MAX_PX} | p=0.5"
log "  Output dir : ${OUTPUT_DIR}"
log "  Log dir    : ${LOG_DIR}"
log ""

cd "${FRAMEWORK_DIR}"

# ══════════════════════════════════════════════════════════
#  VÒNG LẶP CHÍNH: drug × model
# ══════════════════════════════════════════════════════════
for DRUG in "${DRUGS[@]}"; do
    DRUG_MASTER_LOG="${LOG_DIR}/task${TASK}_${DRUG}_erosion_aug_all_models.txt"
    DRUG_START=$(date +%s)
    DRUG_DONE=0
    DRUG_FAIL=0

    log_drug "Drug=${DRUG} | START | ${TOTAL_MODELS} models"
    > "${DRUG_MASTER_LOG}"

    for MODEL in "${ALL_MODELS[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        BATCH_SIZE=$(get_batch_size "${MODEL}")
        MODEL_LOG="${LOG_DIR}/task${TASK}_${DRUG}_${MODEL}_erosion_aug.txt"
        MODEL_START=$(date +%s)

        log "  ┌── [${RUN_IDX}/${TOTAL_RUNS}] Drug=${DRUG} | Model=${MODEL} | batch=${BATCH_SIZE} | erosion_max=${EROSION_MAX_PX}px"

        cat > "${STATUS_FILE}" <<EOF
RUNNING: [${RUN_IDX}/${TOTAL_RUNS}] Drug=${DRUG} | Model=${MODEL}
Started: $(date '+%Y-%m-%d %H:%M:%S')
Batch  : ${BATCH_SIZE} | BF16: $([ "${USE_BF16}" = "true" ] && echo ON || echo OFF)
ErosionAug: max_px=${EROSION_MAX_PX} | p=0.5
Log    : ${MODEL_LOG}
EOF

        # ── Chạy python ────────────────────────────────────
        RUN_RESULT="DONE"
        if python main.py \
                --task              ${TASK}              \
                --drug              ${DRUG}              \
                --model             ${MODEL}             \
                --seeds             ${SEEDS}             \
                --batch_size        ${BATCH_SIZE}        \
                --threshold_method  ${THRESHOLD_METHOD}  \
                --best_metric       ${BEST_METRIC}       \
                --best_metric_level ${BEST_METRIC_LEVEL} \
                --loss              ${LOSS}              \
                --output_dir        "${OUTPUT_DIR}"      \
                --mask_erosion_aug                       \
                --mask_erosion_max_px ${EROSION_MAX_PX}  \
                ${BF16_FLAG}                             \
                2>&1 | tee "${MODEL_LOG}" | tee -a "${DRUG_MASTER_LOG}"; then

            RUN_RESULT="DONE"
            DRUG_DONE=$((DRUG_DONE + 1))

        else
            # ── OOM → retry với batch/2 ────────────────────
            if grep -q "CUDA out of memory\|OutOfMemoryError" "${MODEL_LOG}" 2>/dev/null; then
                HALF_BATCH=$(( BATCH_SIZE / 2 ))
                log "  │   ⚠ OOM! Retry batch=${BATCH_SIZE} → ${HALF_BATCH}"

                if python main.py \
                        --task              ${TASK}              \
                        --drug              ${DRUG}              \
                        --model             ${MODEL}             \
                        --seeds             ${SEEDS}             \
                        --batch_size        ${HALF_BATCH}        \
                        --threshold_method  ${THRESHOLD_METHOD}  \
                        --best_metric       ${BEST_METRIC}       \
                        --best_metric_level ${BEST_METRIC_LEVEL} \
                        --loss              ${LOSS}              \
                        --output_dir        "${OUTPUT_DIR}"      \
                        --mask_erosion_aug                       \
                        --mask_erosion_max_px ${EROSION_MAX_PX}  \
                        ${BF16_FLAG}                             \
                        2>&1 | tee "${MODEL_LOG}.retry" | tee -a "${DRUG_MASTER_LOG}"; then

                    cat "${MODEL_LOG}.retry" >> "${MODEL_LOG}"
                    RUN_RESULT="RETRY_OK"
                    DRUG_DONE=$((DRUG_DONE + 1))
                else
                    RUN_RESULT="FAIL_OOM"
                    DRUG_FAIL=$((DRUG_FAIL + 1))
                    FAILED_RUNS+=("${DRUG}/${MODEL}")
                fi
            else
                RUN_RESULT="FAIL"
                DRUG_FAIL=$((DRUG_FAIL + 1))
                FAILED_RUNS+=("${DRUG}/${MODEL}")
            fi
        fi

        MODEL_END=$(date +%s)
        MODEL_SEC=$(( MODEL_END - MODEL_START ))
        MODEL_MIN=$(( MODEL_SEC / 60 ))
        MODEL_REM=$(( MODEL_SEC % 60 ))

        RUN_STATUS["${DRUG}/${MODEL}"]="${RUN_RESULT}"
        RUN_DURATION["${DRUG}/${MODEL}"]="${MODEL_SEC}"

        case "${RUN_RESULT}" in
            DONE)       STATUS_ICON="✓" ;;
            RETRY_OK)   STATUS_ICON="↺" ;;
            *)          STATUS_ICON="✗" ;;
        esac

        log "  └── ${STATUS_ICON} ${MODEL} | ${RUN_RESULT} | ${MODEL_MIN}m${MODEL_REM}s"

    done  # end model loop

    DRUG_END=$(date +%s)
    DRUG_SEC=$(( DRUG_END - DRUG_START ))
    DRUG_H=$(( DRUG_SEC / 3600 ))
    DRUG_M=$(( (DRUG_SEC % 3600) / 60 ))
    DRUG_S=$(( DRUG_SEC % 60 ))

    log ""
    log "  ┌─ TÓM TẮT Drug=${DRUG} ──────────────────────────────"
    log "  │  Thời gian  : ${DRUG_H}h ${DRUG_M}m ${DRUG_S}s"
    log "  │  Thành công : ${DRUG_DONE} / ${TOTAL_MODELS} models"
    log "  │  Thất bại   : ${DRUG_FAIL} models"
    log "  │  Master log : ${DRUG_MASTER_LOG}"
    log "  └───────────────────────────────────────────────────"

    log ""
    log "  Model status (Drug=${DRUG}):"
    for MODEL in "${ALL_MODELS[@]}"; do
        KEY="${DRUG}/${MODEL}"
        ST="${RUN_STATUS[$KEY]:-N/A}"
        SEC="${RUN_DURATION[$KEY]:-0}"
        MIN=$(( SEC / 60 ))
        REM=$(( SEC % 60 ))
        case "${ST}" in
            DONE)     ICON="✓" ;;
            RETRY_OK) ICON="↺" ;;
            *)        ICON="✗" ;;
        esac
        printf "  %s %-28s %-10s %dm%ds\n" "${ICON}" "${MODEL}" "${ST}" "${MIN}" "${REM}" \
            | tee -a "${PROGRESS_LOG}"
    done

done  # end drug loop

# ══════════════════════════════════════════════════════════
#  TỔNG KẾT CUỐI
# ══════════════════════════════════════════════════════════
SCRIPT_END=$(date +%s)
TOTAL_SEC=$(( SCRIPT_END - SCRIPT_START ))
TOTAL_H=$(( TOTAL_SEC / 3600 ))
TOTAL_M=$(( (TOTAL_SEC % 3600) / 60 ))
TOTAL_S=$(( TOTAL_SEC % 60 ))
TOTAL_DONE=$(( TOTAL_RUNS - ${#FAILED_RUNS[@]} ))

log ""
log "╔══════════════════════════════════════════════════════════╗"
log "║  TẤT CẢ XONG                                            ║"
log "╚══════════════════════════════════════════════════════════╝"
log "  Thời gian tổng : ${TOTAL_H}h ${TOTAL_M}m ${TOTAL_S}s"
log "  Thành công     : ${TOTAL_DONE} / ${TOTAL_RUNS} runs"
log "  Thất bại       : ${#FAILED_RUNS[@]} runs"
log "  Output dir     : ${OUTPUT_DIR}"
log "  Progress log   : ${PROGRESS_LOG}"
log ""
log "  Sau khi xong, so sánh baseline vs erosion_aug:"
log "    python antibiotics_framework/parse_log.py \\"
log "      --from-results results --task 1 --drug CIP --out baseline_CIP.xlsx --both"
log "    python antibiotics_framework/parse_log.py \\"
log "      --from-results results_erosion_aug --task 1 --drug CIP --out erosion_aug_CIP.xlsx --both"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    log ""
    log "  ✗ CÁC RUN THẤT BẠI:"
    for FAIL in "${FAILED_RUNS[@]}"; do
        log "    - ${FAIL}"
    done
    echo "FINISHED_WITH_ERRORS | $(date '+%Y-%m-%d %H:%M:%S') | Failed: ${#FAILED_RUNS[@]}/${TOTAL_RUNS}" > "${STATUS_FILE}"
    exit 1
else
    log ""
    log "  ✓ Tất cả ${TOTAL_RUNS} runs thành công!"
    echo "FINISHED_OK | $(date '+%Y-%m-%d %H:%M:%S') | ${TOTAL_RUNS}/${TOTAL_RUNS} | ${TOTAL_H}h${TOTAL_M}m${TOTAL_S}s" > "${STATUS_FILE}"
fi
