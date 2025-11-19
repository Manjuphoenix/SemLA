#!/bin/bash

# usage:
# bash run_all_lora.sh

declare -A CONFIGS

# ==== DEFINE ALL LORA ADAPTERS HERE ==========
# CONFIGS["acdc_conv-rain"]="configs/acdc/rain/lora-rain-acdc.yaml"
# CONFIGS["acdc_conv-snow"]="configs/acdc/snow/lora-snow-acdc.yaml"
# CONFIGS["acdc_conv-fog"]="configs/acdc/fog/lora-fog-acdc.yaml"
# CONFIGS["acdc_conv-night"]="configs/acdc/night/lora-night-acdc.yaml"
# =============================================

# CONFIGS["indraeye"]="configs/indraeye/rgb/indraeye-rgb.yaml" 
# CONFIGS["cs_conv-normal"]="configs/cityscapes/cityscapes_base.yaml"
# CONFIGS["bdd_conv"]="configs/bdd/bdd.yaml"
# CONFIGS["idd_conv"]="configs/idd/idd.yaml"

# # =============================================

# CONFIGS["ade"]="configs/a150/a150.yaml"

# =============================================


CONFIGS["mv_conv"]="configs/mv/mv.yaml"
CONFIGS["msrs_conv"]="configs/msrs/rgb/msrs-rgb.yaml"
CONFIGS["cartr"]="configs/cart/rgb/cart-rgb.yaml"
CONFIGS["openearthmap"]="configs/openearth/openearth.yaml"


# =============================================
# CONFIGS["idd_conv"]="configs/idd/idd.yaml"
# CONFIGS["muses-clear-day"]="configs/muses/clear/muses-clear-day.yaml"
# CONFIGS["muses-clear-night"]="configs/muses/clear/muses-clear-night.yaml"
# CONFIGS["muses-rain-day"]="configs/muses/rain/muses-rain-day.yaml"
# CONFIGS["muses-fog-day"]="configs/muses/fog/muses-fog-day.yaml"
# CONFIGS["muses-fog-night"]="configs/muses/fog/muses-fog-night.yaml"
# CONFIGS["muses-snow-night"]="configs/muses/snow/muses-snow-night.yaml"
# CONFIGS["muses-snow-day"]="configs/muses/snow/muses-snow-day.yaml"



echo "=========================================="
echo " LoRA Adaptors that will be trained now:"
echo "------------------------------------------"
for NAME in "${!CONFIGS[@]}"; do
    echo " - ${NAME}   (${CONFIGS[$NAME]})"
done
echo "=========================================="
echo ""
sleep 2  


for NAME in "${!CONFIGS[@]}"; do

    CONFIG=${CONFIGS[$NAME]}
    OUTDIR="output/${NAME}"

    mkdir -p "${OUTDIR}"

    echo "----------------------------------------------------"
    echo "Running ${NAME}"
    echo "Config: ${CONFIG}"
    echo "Output: ${OUTDIR}"
    echo "----------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 \
    sh run_lora.sh ${CONFIG} 1 ${OUTDIR} MODEL.LORA.NAME ${NAME} #SOLVER.MAX_ITER 5000

done

