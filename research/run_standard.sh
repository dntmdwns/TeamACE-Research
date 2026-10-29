#!/usr/bin/env bash
# Standard experiment runner: two-scene continual learning (scene A, then scene B) with scene-A feature replay,
# using the team standard setup (CONTRIBUTING.md) unless an option says otherwise. Every option below has the
# standard value as its default; SETUP.md lists every option that differs from the standard.
#
#   research/run_standard.sh launch --out DIR [options]    # check prerequisites, then start tmux, one window per GPU
#   research/run_standard.sh check  --out DIR [options]    # the same checks and plan, without starting anything
#   research/run_standard.sh summarize --out DIR           # rebuild DIR/summary.{md,csv,json} from finished runs
#   research/run_standard.sh worker --out DIR --gpu G --seeds S1,S2   # what each tmux window runs (internal)
#
# Per seed: stage A trains a fresh head on scene A (stage_a/seed<N>/, with a JSON record of its settings next to
# it, including sha256 fingerprints of the training code), or reuses --stage_a_head after checking its record and
# sha256. Then, for every method and K: export K
# scene-A features chosen by the method (single-threaded CPU), train stage B on scene B from the stage-A head with
# that replay, and evaluate both scenes. Method `none` is stage B without replay. All of a seed's runs use the same
# stage-A head (its sha256 is checked before every use). Seeds are queued round-robin over --gpus; each GPU runs
# its seeds one after another. Records provenance for each run.
set -uo pipefail

SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
REPO=$(git -C "$(dirname "$SCRIPT")" rev-parse --show-toplevel) || exit 1
HELPER=$REPO/research/standard_runner.py
METHODS_ALLOWED=(none random filtered_random quality voxel voxel_quality fps voxel_weighted)

# Defaults: the team standard.
OUT=""
PYTHON=python
DATASETS_DIR=$REPO/datasets
ENCODER=$REPO/ace_encoder_pretrained.pt
POSE_VERSION=pgt
SCENES=chess,fire
SEEDS=2089,2090,2091
GPUS=0,1,2
METHODS=filtered_random
KS=5000
TRAINING_BUFFER_SIZE=4000000
BATCH_SIZE=5120
EPOCHS=16
PRECISION=legacy
FEATURE_ENCODING=fp16
REPLAY_RATIO_PERCENT=10
BATCH_MODE=stratified
REPROJECTION_FILTER_PX=5
DROP_REPEATED_CANDIDATES=1
VOXEL_SIZE_M=0.25
VOXEL_ALPHAS=1
EXPORT_THREADS=1
OFFSET_A=""
OFFSET_B=""
HEAD_CENTER=scene_a
FP32_COORDINATES=0
STAGE_A_HEAD=""
STAGE_A_HEAD_SHA256=""
ALLOW_HEAD_MISMATCH=0
HEAD_MISMATCH_REASON=""
EXPORT_ONLY=0
SESSION=""
MAX_GPU_MEM_USED_MIB=500

usage() {
    cat <<EOF
usage: $0 launch|check --out DIR [options]
       $0 summarize --out DIR
Options (default in brackets; the defaults are the team standard):
  --out DIR                      output folder; must not exist [required]
  --python PATH                  Python of the ACE environment [$PYTHON]
  --datasets_dir DIR             folder holding <prefix><scene> dataset folders [$DATASETS_DIR]
  --encoder PATH                 pretrained ACE encoder [$ENCODER]
  --pose_version pgt|calibrated  dataset prefix pgt_7scenes_ or 7scenes_; checked against the data [$POSE_VERSION]
  --scenes A,B                   scene order: train A, then B with A replay [$SCENES]
  --seeds LIST                   [$SEEDS]
  --gpus LIST                    physical GPU indices (nvidia-smi); seeds are queued round-robin [$GPUS]
  --methods LIST                 ${METHODS_ALLOWED[*]} [$METHODS]
  --ks LIST                      stored features K per method [$KS]
  --training_buffer_size N       buffer entries per stage [$TRAINING_BUFFER_SIZE]
  --batch_size N                 [$BATCH_SIZE]
  --epochs N                     [$EPOCHS]
  --precision legacy|fp32        [$PRECISION]
  --feature_encoding fp16|fp32|int8   replay store features [$FEATURE_ENCODING]
  --replay_ratio_percent R       share of the stage-B buffer replaced by replay [$REPLAY_RATIO_PERCENT]
  --batch_mode stratified|shuffle     [$BATCH_MODE]
  --reprojection_filter_px PX    L1 reprojection filter for filtered methods [$REPROJECTION_FILTER_PX]
  --drop_repeated_candidates 0|1 de-duplicate the candidate pool (--replay_drop_repeated_candidates) [$DROP_REPEATED_CANDIDATES]
  --voxel_size_m M               voxel size for voxel methods [$VOXEL_SIZE_M]
  --voxel_alphas LIST            voxel_weighted only: one run per alpha (--replay_voxel_alpha), folders
                                 voxel_weighted_alpha<alpha>/K<K>; voxels drawn with probability proportional to
                                 (remaining candidates)^alpha, 0 = equal per voxel, 1 = as filtered_random [$VOXEL_ALPHAS]
  --export_threads N             OMP/MKL threads for the export step [$EXPORT_THREADS]
  --offset_a X,Y,Z               shift scene A's camera positions by this offset in meters, rotation unchanged
                                 (train_ace.py/test_ace.py --scene_offset) [off]
  --offset_b X,Y,Z               the same for scene B [off]
  --head_center scene_a|midpoint|X,Y,Z   center of the fresh stage-A head: scene A's mean camera position (as
                                 ACE), the midpoint of both scenes' (shifted) mean camera positions, or explicit;
                                 stage B keeps the stage-A head's center [$HEAD_CENTER]
  --fp32_coordinates 0|1         head's final layer and center addition in fp32 without TF32, heads saved in fp32
                                 (--fp32_coordinates) [$FP32_COORDINATES]
  --stage_a_head PATH            reuse existing stage-A heads; {seed} is replaced by the seed. The JSON record next
                                 to each head must match this run [train one per seed]
  --stage_a_head_sha256 S=H,...  expected sha256 per seed, for heads without a record (needs --allow_head_mismatch)
  --allow_head_mismatch          use a reused head whose record differs or is missing; recorded in SETUP.md
  --head_mismatch_reason TEXT    why the mismatch is acceptable; written into SETUP.md (needs --allow_head_mismatch)
  --export_only                  stop after each export (no stage-B training or evaluation)
  --session NAME                 tmux session [std-<basename of --out>]
  --max_gpu_mem_used_mib N       a GPU using more memory than this counts as busy [$MAX_GPU_MEM_USED_MIB]
EOF
}

die() { echo "error: $*" >&2; exit 1; }

parse_options() {
    while (($#)); do
        case $1 in
            --allow_head_mismatch) ALLOW_HEAD_MISMATCH=1; shift ;;
            --export_only) EXPORT_ONLY=1; shift ;;
            --out|--python|--datasets_dir|--encoder|--pose_version|--scenes|--seeds|--gpus|--methods|--ks|\
            --training_buffer_size|--batch_size|--epochs|--precision|--feature_encoding|--replay_ratio_percent|\
            --batch_mode|--reprojection_filter_px|--drop_repeated_candidates|--voxel_size_m|--voxel_alphas|--export_threads|\
            --offset_a|--offset_b|--head_center|--fp32_coordinates|--stage_a_head|--stage_a_head_sha256|--head_mismatch_reason|\
            --session|--max_gpu_mem_used_mib)
                (($# >= 2)) || die "$1 needs a value"
                local name=${1#--}; printf -v "${name^^}" '%s' "$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) usage >&2; die "unknown option: $1" ;;
        esac
    done
}

# Setting variables saved to settings.env for the workers.
SETTING_VARS=(OUT PY DATASETS_DIR ENCODER POSE_VERSION SCENES SEEDS GPUS METHODS KS TRAINING_BUFFER_SIZE BATCH_SIZE
              EPOCHS PRECISION FEATURE_ENCODING REPLAY_RATIO_PERCENT BATCH_MODE REPROJECTION_FILTER_PX
              DROP_REPEATED_CANDIDATES VOXEL_SIZE_M VOXEL_ALPHAS EXPORT_THREADS OFFSET_A OFFSET_B HEAD_CENTER FP32_COORDINATES
              STAGE_A_HEAD STAGE_A_HEAD_SHA256 ALLOW_HEAD_MISMATCH HEAD_MISMATCH_REASON EXPORT_ONLY SESSION SCENE_A SCENE_B SCENE_A_DIR
              SCENE_B_DIR CENTER_A CENTER_B HEAD_CENTER_XYZ)

is_list_of_ints() { [[ $1 =~ ^[0-9]+(,[0-9]+)*$ ]]; }
is_xyz() { [[ $1 =~ ^(-?[0-9]+(\.[0-9]*)?,){2}-?[0-9]+(\.[0-9]*)?$ ]]; }
has_duplicates() { [[ -n $(tr , '\n' <<< "$1" | sort | uniq -d) ]]; }

validate() {
    [[ -n $OUT ]] || die "--out is required"
    OUT=$(readlink -m "$OUT")
    [[ -e $OUT ]] && die "refusing to overwrite $OUT (choose a new --out)"
    PY=$(command -v "$PYTHON") || die "python not found: $PYTHON (pass --python)"
    DATASETS_DIR=$(readlink -m "$DATASETS_DIR"); ENCODER=$(readlink -m "$ENCODER")
    [[ -n $STAGE_A_HEAD ]] && STAGE_A_HEAD=$(readlink -m "$STAGE_A_HEAD")
    case $POSE_VERSION in pgt) PREFIX=pgt_7scenes_ ;; calibrated) PREFIX=7scenes_ ;; *) die "--pose_version: pgt or calibrated" ;; esac
    IFS=, read -ra SCENE_LIST <<< "$SCENES"
    ((${#SCENE_LIST[@]} == 2)) && [[ ${SCENE_LIST[0]} != "${SCENE_LIST[1]}" ]] || die "--scenes takes two different scenes, e.g. chess,fire"
    SCENE_A=${SCENE_LIST[0]} SCENE_B=${SCENE_LIST[1]}
    SCENE_A_DIR=$DATASETS_DIR/$PREFIX$SCENE_A SCENE_B_DIR=$DATASETS_DIR/$PREFIX$SCENE_B
    local list
    for list in SEEDS GPUS KS; do
        is_list_of_ints "${!list}" || die "--${list,,} takes comma-separated integers"
        has_duplicates "${!list}" && die "--${list,,} has duplicates"
    done
    has_duplicates "$METHODS" && die "--methods has duplicates"
    IFS=, read -ra SEED_LIST <<< "$SEEDS"; IFS=, read -ra GPU_LIST <<< "$GPUS"
    IFS=, read -ra METHOD_LIST <<< "$METHODS"; IFS=, read -ra K_LIST <<< "$KS"
    [[ $VOXEL_ALPHAS =~ ^[0-9]+(\.[0-9]+)?(,[0-9]+(\.[0-9]+)?)*$ ]] || die "--voxel_alphas takes comma-separated non-negative numbers"
    has_duplicates "$VOXEL_ALPHAS" && die "--voxel_alphas has duplicates"
    [[ $VOXEL_ALPHAS != 1 && " ${METHOD_LIST[*]} " != *" voxel_weighted "* ]] && die "--voxel_alphas applies to method voxel_weighted only"
    IFS=, read -ra ALPHA_LIST <<< "$VOXEL_ALPHAS"
    local method k
    for method in "${METHOD_LIST[@]}"; do
        [[ " ${METHODS_ALLOWED[*]} " == *" $method "* ]] || die "unknown method $method (allowed: ${METHODS_ALLOWED[*]})"
        [[ $method == random && $DROP_REPEATED_CANDIDATES == 1 ]] && die "random (unfiltered) needs --drop_repeated_candidates 0"
        [[ $method == none && $EXPORT_ONLY == 1 ]] && die "method none has nothing to export; drop it with --export_only"
    done
    for k in "${K_LIST[@]}"; do ((k > 0 && k <= TRAINING_BUFFER_SIZE)) || die "K=$k must be in 1..$TRAINING_BUFFER_SIZE"; done
    for var in TRAINING_BUFFER_SIZE BATCH_SIZE EPOCHS EXPORT_THREADS MAX_GPU_MEM_USED_MIB; do
        [[ ${!var} =~ ^[1-9][0-9]*$ ]] || die "--${var,,} takes a positive integer"
    done
    [[ $PRECISION == legacy || $PRECISION == fp32 ]] || die "--precision: legacy or fp32"
    [[ $PRECISION == fp32 && $FEATURE_ENCODING != fp32 ]] && die "--precision fp32 stores fp32 features; pass --feature_encoding fp32"
    [[ $FEATURE_ENCODING =~ ^(fp16|fp32|int8)$ ]] || die "--feature_encoding: fp16, fp32 or int8"
    [[ $BATCH_MODE == stratified || $BATCH_MODE == shuffle ]] || die "--batch_mode: stratified or shuffle"
    [[ $DROP_REPEATED_CANDIDATES == 0 || $DROP_REPEATED_CANDIDATES == 1 ]] || die "--drop_repeated_candidates: 0 or 1"
    [[ -z $OFFSET_A ]] || is_xyz "$OFFSET_A" || die "--offset_a takes X,Y,Z in meters"
    [[ -z $OFFSET_B ]] || is_xyz "$OFFSET_B" || die "--offset_b takes X,Y,Z in meters"
    [[ $HEAD_CENTER == scene_a || $HEAD_CENTER == midpoint ]] || is_xyz "$HEAD_CENTER" || die "--head_center: scene_a, midpoint or X,Y,Z"
    [[ $FP32_COORDINATES == 0 || $FP32_COORDINATES == 1 ]] || die "--fp32_coordinates: 0 or 1"
    [[ -n $STAGE_A_HEAD && $HEAD_CENTER != scene_a ]] && die "--head_center applies to a fresh stage-A head; a reused head keeps its center"
    if [[ -n $STAGE_A_HEAD && ${#SEED_LIST[@]} -gt 1 && $STAGE_A_HEAD != *"{seed}"* ]]; then
        die "--stage_a_head must contain {seed} when running several seeds (each seed has its own head)"
    fi
    [[ -n $STAGE_A_HEAD_SHA256 && -z $STAGE_A_HEAD ]] && die "--stage_a_head_sha256 needs --stage_a_head"
    [[ -n $HEAD_MISMATCH_REASON && $ALLOW_HEAD_MISMATCH != 1 ]] && die "--head_mismatch_reason needs --allow_head_mismatch"
    SESSION=${SESSION:-std-$(basename "$OUT")}
    SESSION=${SESSION//[.:]/_}
}

gpu_for_seed() {   # round-robin queue: the i-th seed runs on GPU i mod (number of GPUs)
    local i
    for i in "${!SEED_LIST[@]}"; do [[ ${SEED_LIST[$i]} == "$1" ]] && { echo "${GPU_LIST[i % ${#GPU_LIST[@]}]}"; return; }; done
}

expected_sha_for_seed() {
    local pair
    IFS=, read -ra pairs <<< "$STAGE_A_HEAD_SHA256"
    for pair in "${pairs[@]}"; do [[ ${pair%%=*} == "$1" ]] && echo "${pair#*=}"; done
}

head_path() { [[ -n $STAGE_A_HEAD ]] && echo "${STAGE_A_HEAD//\{seed\}/$1}" || echo "$OUT/stage_a/seed$1/stage_a_head.pt"; }

# Checks, settings and head checks in a staging folder that becomes --out on success (launch) or is removed (check).
prepare() {
    local mode=$1 problems=0 seed gpu names
    [[ -z $(git -C "$REPO" status --porcelain --untracked-files=no) ]] || die "tracked changes in $REPO; commit them first"
    [[ $mode == launch ]] && command -v tmux >/dev/null && tmux has-session -t "=$SESSION" 2>/dev/null && die "tmux session $SESSION already exists"
    "$PY" "$HELPER" prereqs --python "$PY" --datasets_dir "$DATASETS_DIR" --pose_version "$POSE_VERSION" \
        --scenes "$SCENES" --gpus "$GPUS" --encoder "$ENCODER" --max_gpu_mem_used_mib "$MAX_GPU_MEM_USED_MIB" || exit 1
    names=$(for gpu in "${GPU_LIST[@]}"; do nvidia-smi --id="$gpu" --query-gpu=name --format=csv,noheader; done | sort -u)
    (( $(wc -l <<< "$names") == 1 )) || die "--gpus mixes GPU models ($(paste -sd, <<< "$names")); results are only comparable on one model"
    # Mean training camera position of each (shifted) scene, as the dataset computes it: the region centers of the
    # evaluation diagnostic, and the source of --head_center midpoint.
    CENTER_A=$("$PY" "$HELPER" scene-center --dataset "$SCENE_A_DIR" "--offset=$OFFSET_A") || exit 1
    CENTER_B=$("$PY" "$HELPER" scene-center --dataset "$SCENE_B_DIR" "--offset=$OFFSET_B") || exit 1
    case $HEAD_CENTER in
        scene_a) HEAD_CENTER_XYZ="" ;;   # the dataset's own mean, exactly as without the option
        midpoint) HEAD_CENTER_XYZ=$("$PY" "$HELPER" midpoint "--a=$CENTER_A" "--b=$CENTER_B") || exit 1 ;;
        *) HEAD_CENTER_XYZ=$HEAD_CENTER ;;
    esac
    mkdir -p "$(dirname "$OUT")"
    STAGING=$(mktemp -d "$(dirname "$OUT")/.$(basename "$OUT").staging.XXXX") || exit 1
    trap 'rm -rf "$STAGING"' EXIT
    "$PY" "$HELPER" write-settings "$STAGING/settings.json" out="$OUT" repo="$REPO" python="$PY" \
        datasets_dir="$DATASETS_DIR" \
        encoder="$ENCODER" encoder_sha256="$(sha256sum "$ENCODER" | cut -d' ' -f1)" pose_version="$POSE_VERSION" \
        scenes="$SCENES" scene_a_dir="$SCENE_A_DIR" scene_b_dir="$SCENE_B_DIR" seeds="$SEEDS" gpus="$GPUS" \
        methods="$METHODS" ks="$KS" training_buffer_size="$TRAINING_BUFFER_SIZE" batch_size="$BATCH_SIZE" \
        epochs="$EPOCHS" precision="$PRECISION" feature_encoding="$FEATURE_ENCODING" \
        replay_ratio_percent="$REPLAY_RATIO_PERCENT" batch_mode="$BATCH_MODE" \
        reprojection_filter_px="$REPROJECTION_FILTER_PX" drop_repeated_candidates="$DROP_REPEATED_CANDIDATES" \
        voxel_size_m="$VOXEL_SIZE_M" voxel_alphas="$VOXEL_ALPHAS" export_threads="$EXPORT_THREADS" offset_a="$OFFSET_A" offset_b="$OFFSET_B" \
        head_center="$HEAD_CENTER" head_center_xyz="$HEAD_CENTER_XYZ" fp32_coordinates="$FP32_COORDINATES" \
        center_a="$CENTER_A" center_b="$CENTER_B" stage_a_head="$STAGE_A_HEAD" \
        allow_head_mismatch="$ALLOW_HEAD_MISMATCH" head_mismatch_reason="$HEAD_MISMATCH_REASON" export_only="$EXPORT_ONLY" session="$SESSION" \
        host="$(hostname)" git_branch="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" \
        git_commit="$(git -C "$REPO" rev-parse HEAD)" runner_sha256="$(sha256sum "$SCRIPT" | cut -d' ' -f1)" \
        helper_sha256="$(sha256sum "$HELPER" | cut -d' ' -f1)" invocation="$INVOCATION" created_utc="$(date -u +%FT%TZ)" || exit 1
    "$PY" "$HELPER" env --gpu "${GPU_LIST[0]}" > "$STAGING/environment.json" || exit 1
    if [[ -n $STAGE_A_HEAD ]]; then
        echo "Checking reused stage-A heads against this run's settings:"
        for seed in "${SEED_LIST[@]}"; do
            mkdir -p "$STAGING/stage_a/seed$seed"
            local override=(); [[ $ALLOW_HEAD_MISMATCH == 1 ]] && override=(--allow_mismatch)
            "$PY" "$HELPER" head-check --settings "$STAGING/settings.json" --seed "$seed" --gpu "$(gpu_for_seed "$seed")" \
                --head "$(head_path "$seed")" --expected_sha256 "$(expected_sha_for_seed "$seed")" \
                --out "$STAGING/stage_a/seed$seed/head_check.json" "${override[@]}" || problems=1
            echo "  seed $seed: $(json_get "$STAGING/stage_a/seed$seed/head_check.json" status) ($(head_path "$seed"))"
        done
        ((problems == 0)) || die "refusing to reuse the stage-A heads above (fix them, or pass --allow_head_mismatch to record the deviation)"
    fi
    { declare -p "${SETTING_VARS[@]}"; declare -p SEED_LIST GPU_LIST METHOD_LIST K_LIST ALPHA_LIST; } > "$STAGING/settings.env"
    "$PY" "$HELPER" setup "$STAGING" >/dev/null || exit 1
    echo
    cat "$STAGING/SETUP.md"
    echo "Plan:"
    for gpu in "${GPU_LIST[@]}"; do
        local queued=()
        for seed in "${SEED_LIST[@]}"; do [[ $(gpu_for_seed "$seed") == "$gpu" ]] && queued+=("$seed"); done
        ((${#queued[@]})) && echo "  GPU $gpu: seeds ${queued[*]}, each: stage A, then ${METHOD_LIST[*]} x K ${K_LIST[*]}" \
            "$([[ " ${METHOD_LIST[*]} " == *" voxel_weighted "* ]] && echo "(voxel_weighted x alpha ${ALPHA_LIST[*]})")"
    done
}

# ------------------------------------------------------------------------------------------------------- worker

# Run one command: append it to commands.txt, log stdout+stderr, fail if the command fails.
step() {
    local log=$1; shift
    printf '%q ' "$@" >> "$RUN_DIR/commands.txt"; printf '> %s 2>&1\n' "$log" >> "$RUN_DIR/commands.txt"
    echo "[$(date -u +%FT%TZ)] $*"
    "$@" 2>&1 | tee "$RUN_DIR/logs/$log"
}

# Provenance for one run directory.
write_provenance() {
    local seed=$1 gpu=$2 experiment=$3
    mkdir -p "$RUN_DIR/logs"
    {
        echo "experiment: $experiment"
        echo "seed: $seed"
        echo "gpu_index: $gpu ($(nvidia-smi --id="$gpu" --query-gpu=name --format=csv,noheader))"
        echo "host: $(hostname)"
        echo "started_utc: $(date -u +%FT%TZ)"
        echo "git_branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
        echo "git_commit: $(git -C "$REPO" rev-parse HEAD)"
        echo "git_status_porcelain (tracked):"; git -C "$REPO" status --porcelain --untracked-files=no | sed 's/^/  /'
        echo "runner: $SCRIPT (sha256 $(sha256sum "$SCRIPT" | cut -d' ' -f1))"
        echo "python: $PY"
        "$PY" "$HELPER" env --gpu "$gpu" --provenance
        echo "${SCENE_A}_dataset: $(readlink -f "$SCENE_A_DIR")"
        echo "${SCENE_B}_dataset: $(readlink -f "$SCENE_B_DIR")"
        echo "encoder: $ENCODER"
        if [[ $experiment != stage_a ]]; then
            echo "${SCENE_A}_head ($([[ -n $STAGE_A_HEAD ]] && echo reused || echo "stage A of this invocation")): $(head_path "$seed")"
            echo "${SCENE_A}_head_sha256: $HEAD_SHA"
        fi
        echo "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "export_env: OMP_NUM_THREADS=$EXPORT_THREADS MKL_NUM_THREADS=$EXPORT_THREADS OMP_DYNAMIC=FALSE"
    } > "$RUN_DIR/provenance.txt"
    git -C "$REPO" diff HEAD > "$RUN_DIR/git_diff.patch"
    : > "$RUN_DIR/commands.txt"
}

json_get() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

# Same head for every run of a seed: compare with the sha256 taken when stage A finished.
check_head() {
    local head now
    head=$(head_path "$1")
    now=$(sha256sum "$head" | cut -d' ' -f1)
    [[ $now == "$HEAD_SHA" ]] || { echo "stage-A head changed during the invocation: $head ($now != $HEAD_SHA)"; return 1; }
}

offset_args() {   # offset_args <offset>: --scene_offset for a non-empty offset
    [[ -n $1 ]] && echo "--scene_offset=$1"   # "=" form: argparse would read a leading "-" as an option
}

evaluate() {   # evaluate <head> <stage> <scene> <scene dir> <offset>
    step "test_$2_$3.log" "$PY" "$REPO/test_ace.py" "$4" "$1" --precision "$PRECISION" --encoder_path "$ENCODER" \
        --session "$2_$3" --metrics_json "$RUN_DIR/eval_$2_$3.json" $(offset_args "$5") "${COORD_ARGS[@]}" \
        --region_centers "$SCENE_A=$CENTER_A" "$SCENE_B=$CENTER_B"
}

stage_a() {
    local seed=$1 gpu=$2 head
    head=$(head_path "$seed")
    RUN_DIR=$OUT/stage_a/seed$seed
    write_provenance "$seed" "$gpu" stage_a || return 1
    if [[ -z $STAGE_A_HEAD ]]; then
        # Train a fresh head on scene A and record its settings next to it.
        local center=(); [[ -n $HEAD_CENTER_XYZ ]] && center=("--head_center=$HEAD_CENTER_XYZ")
        step train_stage_a.log "$PY" "$REPO/train_ace.py" "$SCENE_A_DIR" "$head" "${TRAIN_ARGS[@]}" \
            $(offset_args "$OFFSET_A") "${center[@]}" --seed "$seed" \
            --training_metrics_json "$RUN_DIR/train_stage_a.json" || return 1
        step head_record.log "$PY" "$HELPER" head-record --settings "$OUT/settings.json" --seed "$seed" --gpu "$gpu" \
            --head "$head" --dataset "$(readlink -f "$SCENE_A_DIR")" --command "$(tail -n1 "$RUN_DIR/commands.txt" | sed 's/ > .*//')" || return 1
        echo "${SCENE_A}_head (trained): $head" >> "$RUN_DIR/provenance.txt"
    else
        echo "${SCENE_A}_head (reused): $head; check: $(json_get "$RUN_DIR/head_check.json" status)" >> "$RUN_DIR/provenance.txt"
    fi
    HEAD_SHA=$(sha256sum "$head" | cut -d' ' -f1)
    if [[ -n $STAGE_A_HEAD ]]; then   # the head must still be the one checked at launch
        [[ $HEAD_SHA == $(json_get "$RUN_DIR/head_check.json" head_sha256) ]] \
            || { echo "stage-A head changed since launch: $head"; return 1; }
    fi
    echo "${SCENE_A}_head_sha256: $HEAD_SHA" >> "$RUN_DIR/provenance.txt"
    [[ $EXPORT_ONLY == 1 ]] && return 0
    evaluate "$head" stage_a "$SCENE_A" "$SCENE_A_DIR" "$OFFSET_A" || return 1
    evaluate "$head" stage_a "$SCENE_B" "$SCENE_B_DIR" "$OFFSET_B" || return 1
    "$PY" "$HELPER" run-summary "$RUN_DIR" --kind stage_a --scenes "$SCENES" --stage_a_dir "$RUN_DIR"
}

stage_b() {   # stage_b <seed> <method> <K> [<alpha>]; method none: no replay; alpha: voxel_weighted only
    local seed=$1 method=$2 k=$3 alpha=${4:-} head replay=() folder=$2 alpha_args=()
    [[ -n $alpha ]] && { folder=${method}_alpha$alpha; alpha_args=(--replay_voxel_alpha "$alpha"); }
    head=$(head_path "$seed")
    if [[ $method == none ]]; then
        RUN_DIR=$OUT/none/seed$seed
        write_provenance "$seed" "$GPU" stage_b_none || return 1
        check_head "$seed" || return 1
    else
        RUN_DIR=$OUT/$folder/K$k/seed$seed
        write_provenance "$seed" "$GPU" "stage_b_${folder}_K$k" || return 1
        check_head "$seed" || return 1
        local dedup=(); [[ $DROP_REPEATED_CANDIDATES == 1 ]] && dedup=(--replay_drop_repeated_candidates)
        # Rebuild the seed's scene-A buffer with the stage-A head and export K features chosen by $method
        # (no head training), single-threaded on the CPU.
        step export.log env OMP_NUM_THREADS="$EXPORT_THREADS" MKL_NUM_THREADS="$EXPORT_THREADS" OMP_DYNAMIC=FALSE \
            "$PY" "$REPO/train_ace.py" "$SCENE_A_DIR" "$RUN_DIR/unused.pt" "${TRAIN_ARGS[@]}" $(offset_args "$OFFSET_A") --seed "$seed" \
            --resume_head "$head" --export_only --export_replay_buffer "$RUN_DIR/replay_$SCENE_A.pt" \
            --replay_buffer_features "$k" --replay_selection "$method" \
            --replay_reprojection_error_px "$REPROJECTION_FILTER_PX" --replay_voxel_size_m "$VOXEL_SIZE_M" \
            --replay_feature_encoding "$FEATURE_ENCODING" "${dedup[@]}" "${alpha_args[@]}" \
            --training_metrics_json "$RUN_DIR/export.json" || return 1
        if [[ $EXPORT_ONLY == 1 ]]; then
            "$PY" "$HELPER" run-summary "$RUN_DIR" --kind replay --scenes "$SCENES" --stage_a_dir "$OUT/stage_a/seed$seed"
            return
        fi
        check_head "$seed" || return 1
        # Replay overwrites the last REPLAY_RATIO_PERCENT of the stage-B buffer.
        replay=(--replay_buffer "$RUN_DIR/replay_$SCENE_A.pt" --replay_ratio_percent "$REPLAY_RATIO_PERCENT"
                --replay_batch_mode "$BATCH_MODE")
    fi
    step train_stage_b.log "$PY" "$REPO/train_ace.py" "$SCENE_B_DIR" "$RUN_DIR/stage_b_head.pt" "${TRAIN_ARGS[@]}" \
        $(offset_args "$OFFSET_B") --seed "$seed" --resume_head "$head" "${replay[@]}" --training_metrics_json "$RUN_DIR/train_stage_b.json" || return 1
    evaluate "$RUN_DIR/stage_b_head.pt" stage_b "$SCENE_A" "$SCENE_A_DIR" "$OFFSET_A" || return 1
    evaluate "$RUN_DIR/stage_b_head.pt" stage_b "$SCENE_B" "$SCENE_B_DIR" "$OFFSET_B" || return 1
    "$PY" "$HELPER" run-summary "$RUN_DIR" --kind "$([[ $method == none ]] && echo none || echo replay)" \
        --scenes "$SCENES" --stage_a_dir "$OUT/stage_a/seed$seed"
}

finish() {   # finish <label> <status>: record the exit status in the run's provenance
    echo "exit_status: $2  finished_utc: $(date -u +%FT%TZ)" >> "$RUN_DIR/provenance.txt"
    [[ $2 -eq 0 ]] && echo "== $1 OK" || echo "== $1 FAILED ($2)"
}

run_seed() {
    local seed=$1 method k alpha
    local status
    stage_a "$seed" "$GPU"; status=$?
    finish "stage_a seed$seed" $status
    ((status == 0)) || { echo "== skipping seed$seed (no stage-A head)"; return; }
    for method in "${METHOD_LIST[@]}"; do
        if [[ $method == none ]]; then
            stage_b "$seed" none 0; finish "none seed$seed" $?
        elif [[ $method == voxel_weighted ]]; then
            for alpha in "${ALPHA_LIST[@]}"; do
                for k in "${K_LIST[@]}"; do
                    stage_b "$seed" "$method" "$k" "$alpha"; finish "${method}_alpha$alpha K$k seed$seed" $?
                done
            done
        else
            for k in "${K_LIST[@]}"; do stage_b "$seed" "$method" "$k"; finish "$method K$k seed$seed" $?; done
        fi
    done
}

worker() {
    local seeds seed
    IFS=, read -ra seeds <<< "$1"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
    COORD_ARGS=(); [[ $FP32_COORDINATES == 1 ]] && COORD_ARGS=(--fp32_coordinates)
    TRAIN_ARGS=(--training_buffer_size "$TRAINING_BUFFER_SIZE" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS"
                --precision "$PRECISION" --encoder_path "$ENCODER" "${COORD_ARGS[@]}")
    cd "$REPO" || exit 1
    for seed in "${seeds[@]}"; do run_seed "$seed"; done
    echo "== GPU $GPU queue done; summary of the runs finished so far:"
    "$PY" "$HELPER" summarize "$OUT"
}

# --------------------------------------------------------------------------------------------------------- main

COMMAND=${1:-}; shift || true
INVOCATION=$(printf '%q ' "$SCRIPT" "$COMMAND" "$@")
case $COMMAND in
    launch|check)
        parse_options "$@"
        validate
        prepare "$COMMAND"
        if [[ $COMMAND == check ]]; then echo "check only: nothing started, $OUT not created"; exit 0; fi
        mv "$STAGING" "$OUT" || exit 1
        trap - EXIT
        mkdir -p "$OUT/logs"
        first=1
        for gpu in "${GPU_LIST[@]}"; do
            queued=()
            for seed in "${SEED_LIST[@]}"; do [[ $(gpu_for_seed "$seed") == "$gpu" ]] && queued+=("$seed"); done
            ((${#queued[@]})) || continue
            seeds=$(IFS=,; echo "${queued[*]}")
            cmd="bash $(printf '%q' "$SCRIPT") worker --out $(printf '%q' "$OUT") --gpu $gpu --seeds $seeds; exec bash"
            if ((first)); then tmux new-session -d -s "$SESSION" -n "gpu$gpu" "$cmd"; first=0
            else tmux new-window -t "$SESSION:" -n "gpu$gpu" "$cmd"; fi
        done
        echo "started tmux session $SESSION (attach: tmux attach -t $SESSION); results in $OUT"
        ;;
    worker)
        OUT=""; GPU=""; WORKER_SEEDS=""
        while (($#)); do
            case $1 in --out) OUT=$2 ;; --gpu) GPU=$2 ;; --seeds) WORKER_SEEDS=$2 ;; *) die "worker: unknown option $1" ;; esac
            shift 2
        done
        source "$OUT/settings.env" || exit 1
        worker "$WORKER_SEEDS" 2>&1 | tee -a "$OUT/logs/gpu$GPU.console.log"
        ;;
    summarize)
        [[ ${1:-} == --out && -n ${2:-} ]] || die "usage: $0 summarize --out DIR"
        source "$2/settings.env" || exit 1
        "$PY" "$HELPER" summarize "$OUT"
        ;;
    *)
        usage; exit 2
        ;;
esac
