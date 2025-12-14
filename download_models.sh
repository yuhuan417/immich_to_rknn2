#!/bin/bash
# Script to download all immich-app models from Hugging Face
# Usage: ./download_models.sh [output_directory]
# Default output directory: ./models
# Generated from: .venv/bin/python -c "from huggingface_hub import list_models; ..."

OUTPUT_DIR="${1:-./models}"
mkdir -p "$OUTPUT_DIR"

echo "Downloading immich-app models to: $OUTPUT_DIR"
echo "=============================================="

# Complete list of all 64 immich-app models on Hugging Face
# Retrieved via huggingface_hub API
MODELS=(
    "immich-app/antelopev2"
    "immich-app/ViT-B-32__openai"
    "immich-app/ViT-B-32__laion2b-s34b-b79k"
    "immich-app/ViT-L-16-SigLIP-256__webli"
    "immich-app/RN50__openai"
    "immich-app/RN50__yfcc15m"
    "immich-app/RN50__cc12m"
    "immich-app/RN101__openai"
    "immich-app/RN101__yfcc15m"
    "immich-app/RN50x4__openai"
    "immich-app/RN50x16__openai"
    "immich-app/RN50x64__openai"
    "immich-app/ViT-B-32__laion2b_e16"
    "immich-app/ViT-B-32__laion400m_e31"
    "immich-app/ViT-B-32__laion400m_e32"
    "immich-app/ViT-B-16__openai"
    "immich-app/ViT-B-16__laion400m_e31"
    "immich-app/ViT-B-16__laion400m_e32"
    "immich-app/ViT-B-16-plus-240__laion400m_e31"
    "immich-app/ViT-B-16-plus-240__laion400m_e32"
    "immich-app/ViT-L-14__openai"
    "immich-app/ViT-L-14__laion400m_e31"
    "immich-app/ViT-L-14__laion400m_e32"
    "immich-app/ViT-L-14__laion2b-s32b-b82k"
    "immich-app/ViT-L-14-336__openai"
    "immich-app/ViT-H-14__laion2b-s32b-b79k"
    "immich-app/ViT-g-14__laion2b-s12b-b42k"
    "immich-app/LABSE-Vit-L-14"
    "immich-app/XLM-Roberta-Large-Vit-B-32"
    "immich-app/XLM-Roberta-Large-Vit-B-16Plus"
    "immich-app/XLM-Roberta-Large-Vit-L-14"
    "immich-app/buffalo_s"
    "immich-app/buffalo_m"
    "immich-app/buffalo_l"
    "immich-app/ViT-L-14-quickgelu__dfn2b"
    "immich-app/ViT-H-14-quickgelu__dfn5b"
    "immich-app/ViT-H-14-378-quickgelu__dfn5b"
    "immich-app/nllb-clip-base-siglip__v1"
    "immich-app/nllb-clip-large-siglip__v1"
    "immich-app/XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k"
    "immich-app/buffalo_l_batch"
    "immich-app/XLM-Roberta-Base-ViT-B-32__laion5b_s13b_b90k"
    "immich-app/nllb-clip-base-siglip__mrl"
    "immich-app/nllb-clip-large-siglip__mrl"
    "immich-app/ViT-B-16-SigLIP-i18n-256__webli"
    "immich-app/ViT-B-16-SigLIP__webli"
    "immich-app/ViT-B-16-SigLIP-256__webli"
    "immich-app/ViT-B-16-SigLIP-384__webli"
    "immich-app/ViT-B-16-SigLIP-512__webli"
    "immich-app/ViT-L-16-SigLIP-384__webli"
    "immich-app/ViT-SO400M-14-SigLIP-384__webli"
    "immich-app/scrfd_34g_gnkps"
    "immich-app/ViT-B-16-SigLIP2__webli"
    "immich-app/ViT-B-32-SigLIP2-256__webli"
    "immich-app/ViT-L-16-SigLIP2-256__webli"
    "immich-app/ViT-L-16-SigLIP2-384__webli"
    "immich-app/ViT-L-16-SigLIP2-512__webli"
    "immich-app/ViT-SO400M-14-SigLIP2-378__webli"
    "immich-app/ViT-SO400M-14-SigLIP2__webli"
    "immich-app/ViT-SO400M-16-SigLIP2-256__webli"
    "immich-app/ViT-SO400M-16-SigLIP2-384__webli"
    "immich-app/ViT-SO400M-16-SigLIP2-512__webli"
    "immich-app/ViT-gopt-16-SigLIP2-256__webli"
    "immich-app/ViT-gopt-16-SigLIP2-384__webli"
)

# Counter for progress
TOTAL=${#MODELS[@]}
CURRENT=0
FAILED=()

for MODEL in "${MODELS[@]}"; do
    ((CURRENT++))
    MODEL_NAME=$(basename "$MODEL")
    echo ""
    echo "[$CURRENT/$TOTAL] Downloading: $MODEL"
    echo "-------------------------------------------"
    
    # Download the model using hf CLI from venv
    if ./hfd.sh "$MODEL" --local-dir "$OUTPUT_DIR/$MODEL_NAME" -x 5 -j 5 --exclude *.armnn *.rknn; then
        echo "✅ Successfully downloaded: $MODEL_NAME"
    else
        echo "❌ Failed to download: $MODEL_NAME"
        FAILED+=("$MODEL")
    fi
done

echo ""
echo "=============================================="
echo "Download complete!"
echo "Total models: $TOTAL"
echo "Successful: $((TOTAL - ${#FAILED[@]}))"
echo "Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for MODEL in "${FAILED[@]}"; do
        echo "  - $MODEL"
    done
    exit 1
fi

echo ""
echo "All models saved to: $OUTPUT_DIR"
