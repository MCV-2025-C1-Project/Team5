# ============================================================
# Makefile for CBIR Project
# ============================================================

PYTHON := python
OUTDIR := data/descriptors
RESULTS := results
VALUES_PER_BIN := 5
METRIC := canberra
K := 10

# ------------------------------------------------------------
# Descriptor Computation
# ------------------------------------------------------------
.PHONY: descriptors
descriptors:
	@echo "Computing HSV descriptors for BBDD..."
	@$(PYTHON) -m src.descriptors.compute_descriptors \
		--descriptor hsv \
		--input data/raw/BBDD \
		--outdir $(OUTDIR) \
		--values_per_bin $(VALUES_PER_BIN)

	@echo "\nComputing LAB descriptors for BBDD..."
	@$(PYTHON) -m src.descriptors.compute_descriptors \
		--descriptor lab \
		--input data/raw/BBDD \
		--outdir $(OUTDIR) \
		--values_per_bin $(VALUES_PER_BIN)

	@echo "\nComputing HSV descriptors for QSD1..."
	@$(PYTHON) -m src.descriptors.compute_descriptors \
		--descriptor hsv \
		--input data/raw/qsd1_w1 \
		--outdir $(OUTDIR) \
		--values_per_bin $(VALUES_PER_BIN)

	@echo "\nComputing LAB descriptors for QSD1..."
	@$(PYTHON) -m src.descriptors.compute_descriptors \
		--descriptor lab \
		--input data/raw/qsd1_w1 \
		--outdir $(OUTDIR) \
		--values_per_bin $(VALUES_PER_BIN)

	@echo "\nComputing HSV descriptors for QST1..."
	@$(PYTHON) -m src.descriptors.compute_descriptors \
		--descriptor hsv \
		--input data/raw/qst1_w1 \
		--outdir $(OUTDIR) \
		--values_per_bin $(VALUES_PER_BIN)

	@echo "\nComputing LAB descriptors for QST1..."
	@$(PYTHON) -m src.descriptors.compute_descriptors \
		--descriptor lab \
		--input data/raw/qst1_w1 \
		--outdir $(OUTDIR) \
		--values_per_bin $(VALUES_PER_BIN)

	@echo "\n✅ Descriptor computation completed successfully!"

# ------------------------------------------------------------
# Find Matching (Retrieval)
# ------------------------------------------------------------
.PHONY: find_matching
find_matching:
	@echo "Running retrieval for HSV (QSD1)..."
	@$(PYTHON) -m src.models.find_matches data/descriptors/qsd1_w1_hsv_vpb5.pkl data/descriptors/BBDD_hsv_vpb5.pkl --metric $(METRIC) --k $(K) --outdir $(RESULTS)

	@echo "\nRunning retrieval for HSV (QST1)..."
	@$(PYTHON) -m src.models.find_matches data/descriptors/qst1_w1_hsv_vpb5.pkl data/descriptors/BBDD_hsv_vpb5.pkl --metric $(METRIC) --k $(K) --outdir $(RESULTS)

	@echo "\nRunning retrieval for LAB (QSD1)..."
	@$(PYTHON) -m src.models.find_matches data/descriptors/qsd1_w1_lab_vpb5.pkl data/descriptors/BBDD_lab_vpb5.pkl --metric $(METRIC) --k $(K) --outdir $(RESULTS)

	@echo "\nRunning retrieval for LAB (QST1)..."
	@$(PYTHON) -m src.models.find_matches data/descriptors/qst1_w1_lab_vpb5.pkl data/descriptors/BBDD_lab_vpb5.pkl --metric $(METRIC) --k $(K) --outdir $(RESULTS)

	@echo "\n✅ All retrieval processes completed successfully!"

# ------------------------------------------------------------
# Combined target
# ------------------------------------------------------------
.PHONY: all
all: descriptors find_matching
	@echo "\nFull pipeline (descriptors + retrieval) completed!"
