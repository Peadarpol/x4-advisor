"""5-Fold Cross-Validation Threshold Calibration for VECTOR_RELEVANCE_THRESHOLD (N=60) using standard library."""

import json
import logging
import math
from pathlib import Path
import random
import sqlite3
import struct
from typing import Dict, List, Tuple

from x4_advisor.config import get_config
from x4_advisor.embeddings.ollama_embedder import OllamaEmbedder
from x4_advisor.storage.db import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibrate_threshold")

# -----------------------------------------------------------------------------
# Calibration Dataset (N=60)
# -----------------------------------------------------------------------------
# 30 Positive In-Domain Queries stratified by length & manifest
POSITIVE_QUERIES = [
    # Short (<50 tokens)
    {"q": "How do escort ship commands work in fleets?", "expected_manifest": "src_man_egosoft_orders_001", "length_bin": "short"},
    {"q": "What is the radar range behavior of patrol ships?", "expected_manifest": "src_man_egosoft_orders_001", "length_bin": "short"},
    {"q": "How to train pilots with basic seminars?", "expected_manifest": "src_man_economy_primer_001", "length_bin": "short"},
    {"q": "What is the primary shipbuilding resource bottleneck?", "expected_manifest": "src_man_economy_primer_001", "length_bin": "short"},
    {"q": "What is the role of fleet commander supply ships?", "expected_manifest": "src_man_fleet2_001", "length_bin": "short"},
    {"q": "What modules are required for a basic silicon station?", "expected_manifest": "src_man_jkninja_station_001", "length_bin": "short"},
    {"q": "How does base station plot cost scale with distance from jump gates?", "expected_manifest": "src_man_jkninja_station_001", "length_bin": "short"},
    {"q": "How do station revenue accounts accumulate profits?", "expected_manifest": "src_man_jkninja_station_001", "length_bin": "short"},
    {"q": "What are the three map levels of trade information?", "expected_manifest": "src_man_ruggedgamer_trading_001", "length_bin": "short"},
    {"q": "Why do Energy Cells and Hull Parts drive universal trade?", "expected_manifest": "src_man_ruggedgamer_trading_001", "length_bin": "short"},

    # Medium (50-200 tokens)
    {"q": "Explain the difference between orders, default behaviors, and fleet assignments in X4.", "expected_manifest": "src_man_egosoft_orders_001", "length_bin": "medium"},
    {"q": "How does station scanning to thirty percent allow players to trade and make money effectively?", "expected_manifest": "src_man_economy_primer_001", "length_bin": "medium"},
    {"q": "When should a player construct a new factory rather than trading existing AI supply?", "expected_manifest": "src_man_economy_primer_001", "length_bin": "medium"},
    {"q": "What are the key differences between fleet assignment roles like attack, defend, and intercept?", "expected_manifest": "src_man_fleet2_001", "length_bin": "medium"},
    {"q": "How do you set up repeat orders behavior to continuously buy and sell wares between two stations?", "expected_manifest": "src_man_jkninja_station_001", "length_bin": "medium"},
    {"q": "How do assigned mining ships supply a station without charging the manager money?", "expected_manifest": "src_man_jkninja_station_001", "length_bin": "medium"},
    {"q": "What pilot skill level is required to unlock advanced auto trade versus local auto trade?", "expected_manifest": "src_man_ruggedgamer_trading_001", "length_bin": "medium"},
    {"q": "How do scrap processors convert raw scrap cubes and ship wrecks into scrap metal?", "expected_manifest": "src_man_scrappy_guide_001", "length_bin": "medium"},
    {"q": "Why do scrap recyclers consume thousands of energy cells per production cycle?", "expected_manifest": "src_man_scrappy_guide_001", "length_bin": "medium"},
    {"q": "What are the two confirmed base-game sectors with high concentrations of natural raw scrap?", "expected_manifest": "src_man_scrappy_guide_001", "length_bin": "medium"},

    # Long (>200 tokens / multi-concept)
    {"q": "Detail the full progression from early-game transport trading to building self-sufficient manufacturing loops.", "expected_manifest": "src_man_reddit_001", "length_bin": "long"},
    {"q": "How do global orders, trade rules, and blacklists protect automated commercial fleets from hostile factions?", "expected_manifest": "src_man_reddit_001", "length_bin": "long"},
    {"q": "What are the common troubleshooting steps when an automated tug ship fails to deposit scrap at a processor?", "expected_manifest": "src_man_scrappy_guide_001", "length_bin": "long"},
    {"q": "How do trade ships assigned to a station calculate their trading range based on manager stars?", "expected_manifest": "src_man_ruggedgamer_trading_001", "length_bin": "long"},
    {"q": "What is the optimal layout and module sequencing when planning a massive self-contained shipyard complex?", "expected_manifest": "src_man_jkninja_station_001", "length_bin": "long"},
    {"q": "How do reactive fleet assignments respond when a protected commander takes fire versus when enemies enter radar?", "expected_manifest": "src_man_egosoft_orders_001", "length_bin": "long"},
    {"q": "Explain how the closed-loop scrap recycling economy provides Hull Parts and Claytronics independently of traditional mineral extraction.", "expected_manifest": "src_man_scrappy_guide_001", "length_bin": "long"},
    {"q": "What are the primary economic bottlenecks that cause universal shipyard stalls in the mid-game?", "expected_manifest": "src_man_economy_primer_001", "length_bin": "long"},
    {"q": "How do you configure automatic trade fleets to distribute excess refined materials across multiple owned stations?", "expected_manifest": "src_man_jkninja_station_001", "length_bin": "long"},
    {"q": "Explain the step by step workflow to establish an autonomous scrapping operation in high sunlight systems.", "expected_manifest": "src_man_scrappy_guide_001", "length_bin": "long"},
]

# 30 Negative Queries (15 DLC, 10 Out-of-Corpus, 5 Distractors)
NEGATIVE_QUERIES = [
    # 15 DLC Queries
    {"q": "What are the best weapon loadouts for the Terran Syn destroyer in Cradle of Humanity?", "type": "dlc"},
    {"q": "How do you build Computronic Substrate and Silicon Carbide in Sol system?", "type": "dlc"},
    {"q": "What is the optimal trading loop for Segaris Pioneer terraforming projects?", "type": "dlc"},
    {"q": "Where can I buy Split Dragon corvette blueprints in Split Vendetta space?", "type": "dlc"},
    {"q": "How do Boron water distribution networks function in Kingdom End sectors?", "type": "dlc"},
    {"q": "What are the stats and weapon slots of the Boron Ray destroyer?", "type": "dlc"},
    {"q": "How do you capture the Erlking battleship in Tides of Avarice?", "type": "dlc"},
    {"q": "What are the protect sector commands for the Riptide Rakers syndicate in Windfall?", "type": "dlc"},
    {"q": "How do you complete the Timelines mission course for the Xenon H battle?", "type": "dlc"},
    {"q": "What are the production materials for Protein Paste and Terran MREs?", "type": "dlc"},
    {"q": "Where is the jump gate connection to Sanctuary of Darkness located?", "type": "dlc"},
    {"q": "How does the Tide mechanic affect solar panel generation in Avarice sectors?", "type": "dlc"},
    {"q": "What is the top travel speed of the Terran Katana corvette?", "type": "dlc"},
    {"q": "Where are the shipyard modules for the Queendom of Boron sold?", "type": "dlc"},
    {"q": "How do you manage Astrid luxury yacht speed boosters?", "type": "dlc"},

    # 10 Out-of-Corpus Base Game Mechanics
    {"q": "What are the exact casino roulette minigame odds on pirate stations?", "type": "out_of_corpus"},
    {"q": "How many eggs does an adult Spacefly lay inside a station aquarium module?", "type": "out_of_corpus"},
    {"q": "What is the exact formula for crafting security decryption keys from craft benches?", "type": "out_of_corpus"},
    {"q": "How do you trigger the hidden developer Easter egg inside the Asteroid Belt anomaly?", "type": "out_of_corpus"},
    {"q": "What is the audio frequency waveform of the Khaak hive hive-mind transmission?", "type": "out_of_corpus"},
    {"q": "How many credits does a passenger pay for luxury taxi transport missions?", "type": "out_of_corpus"},
    {"q": "What is the maximum paint modification inventory capacity in player storage?", "type": "out_of_corpus"},
    {"q": "What are the unlock conditions for the hidden spacesuit blaster modifications?", "type": "out_of_corpus"},
    {"q": "How do you change the interior ambient lighting color of player ship bridges?", "type": "out_of_corpus"},
    {"q": "What is the mathematical drop rate percentage of ancient relic relics from lockboxes?", "type": "out_of_corpus"},

    # 5 Distractor / Irrelevant Queries
    {"q": "How do I install graphics mods in Skyrim Special Edition?", "type": "distractor"},
    {"q": "What is the best build for a Barbarian in Diablo 4 season 3?", "type": "distractor"},
    {"q": "How to make authentic sourdough bread with a wild yeast starter?", "type": "distractor"},
    {"q": "What are the system requirements for running Cyberpunk 2077 with ray tracing?", "type": "distractor"},
    {"q": "Explain the fundamental difference between quantum mechanics and general relativity.", "type": "distractor"},
]


def run_calibration():
    config = get_config(validate=False)
    conn = get_connection(config.database_path)
    embedder = OllamaEmbedder(
        endpoint=config.ollama_endpoint,
        model_name=config.embedding_model,
    )

    logger.info("Collecting unbounded KNN cosine similarity scores for all 60 queries...")

    # Embed and retrieve top-1 similarity for each query
    pos_scores = []
    for item in POSITIVE_QUERIES:
        q_vec = embedder.embed_text(item["q"])
        vec_blob = struct.pack(f"{len(q_vec)}f", *q_vec)
        row = conn.execute(
            """
            SELECT distance FROM knowledge_chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance LIMIT 1
            """,
            (vec_blob,),
        ).fetchone()
        sim = 1.0 - row[0] if row else 0.0
        pos_scores.append({"query": item["q"], "sim": sim, "bin": item["length_bin"], "manifest": item["expected_manifest"]})

    neg_scores = []
    for item in NEGATIVE_QUERIES:
        q_vec = embedder.embed_text(item["q"])
        vec_blob = struct.pack(f"{len(q_vec)}f", *q_vec)
        row = conn.execute(
            """
            SELECT distance FROM knowledge_chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance LIMIT 1
            """,
            (vec_blob,),
        ).fetchone()
        sim = 1.0 - row[0] if row else 0.0
        neg_scores.append({"query": item["q"], "sim": sim, "type": item["type"]})

    conn.close()

    pos_sims = [p["sim"] for p in pos_scores]
    neg_sims = [n["sim"] for n in neg_scores]

    pos_min, pos_max, pos_med = min(pos_sims), max(pos_sims), sorted(pos_sims)[len(pos_sims)//2]
    neg_min, neg_max, neg_med = min(neg_sims), max(neg_sims), sorted(neg_sims)[len(neg_sims)//2]

    logger.info(f"Positive Similarity Range: min={pos_min:.4f}, median={pos_med:.4f}, max={pos_max:.4f}")
    logger.info(f"Negative Similarity Range: min={neg_min:.4f}, median={neg_med:.4f}, max={neg_max:.4f}")

    # -------------------------------------------------------------------------
    # 5-Fold Stratified Cross-Validation
    # -------------------------------------------------------------------------
    n_folds = 5
    fold_taus = []
    fold_f1s = []

    pos_indices = list(range(len(pos_scores)))
    neg_indices = list(range(len(neg_scores)))
    random.seed(42)
    random.shuffle(pos_indices)
    random.shuffle(neg_indices)

    fold_size_pos = len(pos_scores) // n_folds
    fold_size_neg = len(neg_scores) // n_folds

    for i in range(n_folds):
        val_pos_idx = set(pos_indices[i * fold_size_pos:(i + 1) * fold_size_pos])
        val_neg_idx = set(neg_indices[i * fold_size_neg:(i + 1) * fold_size_neg])

        val_pos = [pos_sims[k] for k in val_pos_idx]
        val_neg = [neg_sims[k] for k in val_neg_idx]
        train_pos = [pos_sims[k] for k in range(len(pos_sims)) if k not in val_pos_idx]
        train_neg = [neg_sims[k] for k in range(len(neg_sims)) if k not in val_neg_idx]

        # Candidate thresholds
        t_min = min(train_neg)
        t_max = max(max(train_pos), max(train_neg))
        steps = 200
        candidates = [t_min + step * (t_max - t_min) / steps for step in range(steps + 1)]

        best_tau = 0.50
        best_f1 = 0.0

        for tau in candidates:
            train_fp = sum(1 for s in train_neg if s >= tau)
            train_tp = sum(1 for s in train_pos if s >= tau)
            train_fn = sum(1 for s in train_pos if s < tau)

            if train_fp == 0 and (train_tp + train_fn) > 0:
                prec = 1.0
                rec = train_tp / (train_tp + train_fn)
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = tau

        if best_f1 == 0.0:
            best_tau = max(train_neg) + 0.001

        # Evaluate on validation fold
        val_tp = sum(1 for s in val_pos if s >= best_tau)
        val_fp = sum(1 for s in val_neg if s >= best_tau)
        val_fn = sum(1 for s in val_pos if s < best_tau)
        val_prec = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
        val_rec = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
        val_f1 = (2 * val_prec * val_rec) / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0.0

        fold_taus.append(best_tau)
        fold_f1s.append(val_f1)
        logger.info(f"Fold {i+1}: Best Tau={best_tau:.4f}, Val F1={val_f1:.4f}, Val FP={val_fp}")

    tau_star = sum(fold_taus) / len(fold_taus)
    tau_min = min(fold_taus)
    tau_max = max(fold_taus)

    # Full 30-Negative Confirmation Pass
    full_neg_fp = sum(1 for s in neg_sims if s >= tau_star)
    full_pos_tp = sum(1 for s in pos_sims if s >= tau_star)
    full_prec = full_pos_tp / (full_pos_tp + full_neg_fp) if (full_pos_tp + full_neg_fp) > 0 else 0.0
    full_rec = full_pos_tp / len(pos_sims)
    full_f1 = (2 * full_prec * full_rec) / (full_prec + full_rec) if (full_prec + full_rec) > 0 else 0.0

    # Layer 3 Marginal Band calculation with floor delta = max(1/2 * overlap, 0.03)
    overlap = max(0.0, max(neg_sims) - min(pos_sims))
    delta = max(0.5 * overlap, 0.03)

    logger.info("==================================================")
    logger.info("         CALIBRATION RESULTS (N=60)               ")
    logger.info("==================================================")
    logger.info(f"Calibrated Threshold (tau*): {tau_star:.4f}")
    logger.info(f"Stability Band             : [{tau_min:.4f}, {tau_max:.4f}]")
    logger.info(f"Layer 3 Marginal Band Delta: {delta:.4f} (Band: [{tau_star:.4f}, {tau_star + delta:.4f}])")
    logger.info(f"Full 30-Negative Pass FPs  : {full_neg_fp} / 30")
    logger.info(f"Full Positive Recall       : {full_pos_tp} / 30 ({full_rec:.1%})")
    logger.info(f"Full Calibration F1        : {full_f1:.4f}")
    logger.info("==================================================")

    # Save results artifact
    res_payload = {
        "calibrated_threshold": round(tau_star, 4),
        "stability_band": [round(tau_min, 4), round(tau_max, 4)],
        "layer3_marginal_delta": round(delta, 4),
        "full_negative_fps": full_neg_fp,
        "full_positive_recall": round(full_rec, 4),
        "full_f1": round(full_f1, 4),
        "fold_taus": [round(t, 4) for t in fold_taus],
        "positive_scores": pos_scores,
        "negative_scores": neg_scores,
    }

    out_file = Path("docs/eval/m6_calibration_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res_payload, f, indent=2)
    logger.info(f"Saved calibration results to {out_file}")

    return tau_star, delta


if __name__ == "__main__":
    run_calibration()
