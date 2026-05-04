#!/usr/bin/env python3
"""Generate SONG reproduction technical report as PDF."""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import KeepTogether
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "SONG_reproduction_report.pdf")

# ── styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

BLUE   = colors.HexColor("#1a3a5c")
LBLUE  = colors.HexColor("#3a7abf")
GRAY   = colors.HexColor("#f4f6fa")
DGRAY  = colors.HexColor("#555555")
GREEN  = colors.HexColor("#2e7d32")
RED    = colors.HexColor("#c62828")
MONO   = colors.HexColor("#263238")

def S(name):
    return styles[name]

title_style = ParagraphStyle("title_style",
    parent=styles["Title"],
    fontSize=26, textColor=BLUE, spaceAfter=8, alignment=TA_CENTER, leading=32)

subtitle_style = ParagraphStyle("subtitle_style",
    parent=styles["Normal"],
    fontSize=13, textColor=DGRAY, alignment=TA_CENTER, spaceAfter=4)

h1_style = ParagraphStyle("h1_style",
    parent=styles["Heading1"],
    fontSize=16, textColor=BLUE, spaceBefore=18, spaceAfter=8,
    borderPad=4, leading=20)

h2_style = ParagraphStyle("h2_style",
    parent=styles["Heading2"],
    fontSize=13, textColor=LBLUE, spaceBefore=12, spaceAfter=6, leading=17)

body_style = ParagraphStyle("body_style",
    parent=styles["Normal"],
    fontSize=10.5, leading=16, spaceAfter=6, alignment=TA_JUSTIFY)

bullet_style = ParagraphStyle("bullet_style",
    parent=styles["Normal"],
    fontSize=10.5, leading=15, spaceAfter=3, leftIndent=16,
    bulletIndent=4, bulletFontSize=10)

code_style = ParagraphStyle("code_style",
    parent=styles["Code"],
    fontSize=8.5, leading=12, leftIndent=12, fontName="Courier",
    backColor=GRAY, borderPad=6, spaceAfter=6)

caption_style = ParagraphStyle("caption_style",
    parent=styles["Normal"],
    fontSize=9, textColor=DGRAY, alignment=TA_CENTER,
    spaceBefore=2, spaceAfter=8, italic=True)

def h1(text):    return Paragraph(text, h1_style)
def h2(text):    return Paragraph(text, h2_style)
def body(text):  return Paragraph(text, body_style)
def bul(text):   return Paragraph(f"• {text}", bullet_style)
def cap(text):   return Paragraph(text, caption_style)
def sp(n=6):     return Spacer(1, n)
def hr():        return HRFlowable(width="100%", thickness=0.5,
                                    color=LBLUE, spaceAfter=6, spaceBefore=4)
def code(text):  return Preformatted(text, code_style)

def tbl(data, col_widths, header_row=True):
    t = Table(data, colWidths=col_widths)
    style = [
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, GRAY]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]
    if header_row:
        style += [
            ("BACKGROUND",  (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ]
    t.setStyle(TableStyle(style))
    return t


# ── document ─────────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="SONG Reproduction Report",
        author="Rui Luo",
    )
    story = []

    # ── page 1: title ─────────────────────────────────────────────────────────
    story += [
        Spacer(1, 2.5*cm),
        Paragraph("SONG: Approximate Nearest Neighbor", title_style),
        Paragraph("Search on GPU", title_style),
        Spacer(1, 0.4*cm),
        Paragraph("Reproduction & Technical Deep-Dive Report", subtitle_style),
        Spacer(1, 0.2*cm),
        Paragraph("ICDE 2020 · Fu et al.", subtitle_style),
        Spacer(1, 1.2*cm),
        hr(),
        Spacer(1, 0.5*cm),
        Paragraph("Experiment Configuration", h2_style),
    ]
    cfg = [
        ["Parameter", "Value"],
        ["Dataset",         "SIFT1M (n = 1,000,000 vectors, dim = 128)"],
        ["Queries",         "nq = 1,000"],
        ["Ground truth k",  "k = 10"],
        ["ef_search sweep", "pq ∈ {10, 20, 30, 40, 50}"],
        ["Repeats",         "3 (timing averaged)"],
        ["GPU",             "NVIDIA (cloud, CUDA)"],
        ["EF_CONSTRUCTION", "150  (matches official SONG CONSTRUCT_SEARCH_BUDGET)"],
        ["FIXED_DEGREE",    "31   (matches official SONG)"],
        ["SEARCH_DEGREE",   "15   (matches official SONG)"],
        ["Bloom filter",    "BF_WORDS=256, BF_NHASH=4  (2 KB / block)"],
    ]
    story.append(tbl(cfg, [4*cm, 11.5*cm]))
    story += [
        sp(14),
        Paragraph("Reproduction variants compared:", h2_style),
        bul("<b>Repro CPU Raw</b>  — CPU NSW graph, L2 distance, std::priority_queue search"),
        bul("<b>Repro GPU Raw</b>  — same graph, GPU warp-parallel L2 search kernel"),
        bul("<b>Repro GPU Hash</b> — BitHash projection → Hamming distance NSW, GPU search"),
        bul("<b>Official CPU Raw</b> — official song.cpu binary (wall time)"),
        bul("<b>Official GPU Raw</b> — official song binary (wall time, includes disk I/O)"),
        PageBreak(),
    ]

    # ── page 2: research background ───────────────────────────────────────────
    story += [
        h1("1.  Research Background: GPU-Accelerated ANN"),
        hr(),
        body(
            "Approximate Nearest Neighbor (ANN) search has been a long-standing challenge "
            "in large-scale machine learning, powering applications from image retrieval "
            "to retrieval-augmented generation (RAG). As dataset sizes scaled to hundreds "
            "of millions of vectors, the research community turned to GPUs for acceleration. "
            "The following timeline traces the key milestones."
        ),
        sp(),
        tbl([
            ["Era",         "Representative Work",          "Approach",                         "Limitation"],
            ["Early GPU\nANN",
             "Brute-force\nGPU scan",
             "Parallelize all pairwise distances\nacross GPU cores. 100% recall.",
             "O(n) per query. Infeasible\nat n > 10M."],
            ["LSH / hashing\non GPU",
             "GPU-LSH variants\n(~2012–2016)",
             "Locality-sensitive hashing:\nrandom projections → buckets.\nGPU parallelizes bucket lookup.",
             "Low recall. Requires many\nhash tables. Memory heavy."],
            ["Quantization\non GPU",
             "FAISS-GPU\n(Johnson et al., 2017)",
             "Product Quantization + inverted\nindex. GPU accelerates PQ distance\nand coarse quantizer scan.",
             "Quantization error limits\nrecall. Graph-based methods\nstill outperform on accuracy."],
            ["Graph-based\non GPU\n★ SONG",
             "SONG\n(Fu et al., ICDE 2020)",
             "First system to map NSW\ngraph BFS onto GPU warps.\nWarp-parallel distance computation.",
             "Build is still CPU-side.\nSingle-layer graph limits\nrecall ceiling."],
            ["GPU-native\ngraph design",
             "GGNN (Groh et al., 2021)\nNSG-GPU variants",
             "Graph structure and traversal\nco-designed for GPU memory\naccess patterns.",
             "Still struggles with\nirregular memory access\nat billion scale."],
            ["GPU build +\nsearch",
             "CAGRA\n(Ootomo et al., 2023)\n(cuVS / RAPIDS)",
             "Both graph construction and\nsearch fully on GPU. Optimized\nfor H100 / A100 tensor cores.",
             "Requires high-end GPU.\nComplex implementation."],
            ["Billion-scale /\nmulti-GPU",
             "DiskANN-GPU,\nFAISS billion-scale",
             "SSD-assisted graph traversal,\nmulti-GPU sharding,\nhierarchical index partitioning.",
             "Active research area.\nNo single dominant solution."],
        ], [2.5*cm, 3.5*cm, 5.5*cm, 4*cm]),
        cap("Table 0 — Evolution of GPU-accelerated ANN search. SONG (★) is the starting point of the graph-based GPU branch."),
        sp(),
        body(
            "SONG occupies a pivotal position: it is the <b>first work to demonstrate that "
            "graph-based ANN — the highest-accuracy class of ANN algorithms — can be "
            "efficiently executed on a GPU</b>. Prior GPU ANN work either sacrificed accuracy "
            "(LSH, PQ) or remained CPU-bound (HNSW). Subsequent works (GGNN, CAGRA) "
            "build directly on SONG's insight of warp-level parallelism, extending it to "
            "GPU-native graph construction and billion-scale datasets."
        ),
        PageBreak(),
    ]

    # ── page 3: paper contribution ────────────────────────────────────────────
    story += [
        h1("2.  Paper Contribution & Core Insight"),
        hr(),
        body(
            "The SONG paper (ICDE 2020) addresses <b>approximate nearest neighbor (ANN)</b> "
            "search at scale. Given a base dataset of n vectors and a query, the goal is to "
            "find its k nearest neighbors efficiently, accepting a small recall loss for speed."
        ),
        sp(),
        h2("2.1  Why GPU is Hard for Graph-based ANN"),
        body(
            "State-of-the-art ANN algorithms (HNSW, NSW) use <b>greedy BFS on proximity graphs</b>. "
            "Each BFS step involves:"
        ),
        bul("Reading the current node's neighbor list (irregular memory access)"),
        bul("Computing distances from the query to each neighbor"),
        bul("Updating a priority queue with new candidates"),
        sp(4),
        body(
            "This pattern is <b>sequential by nature</b>: each BFS step depends on the result of "
            "the previous one (which node to expand next). Naively porting this to GPU gives "
            "poor utilization because only one thread does useful work at a time."
        ),
        sp(),
        h2("2.2  SONG's Core Insight"),
        body(
            "SONG observes that within a single BFS step, after fixing the current node cn "
            "and its neighbor list, the <b>distance computation from the query to each neighbor "
            "is completely independent</b>. For a 128-dim L2 distance, this is 128 "
            "multiply-adds — enough arithmetic to keep a full GPU warp busy."
        ),
        sp(4),
        body("The mapping is:"),
        code(
            "  1 Query      →   1 CUDA Block\n"
            "  1 BFS step   →   1 Warp (32 threads) computing 1 distance in parallel\n"
            "  Lane k        →   handles elements [k, k+32, k+64, ...] of the vector\n"
            "  Result        →   butterfly-reduced into lane 0 via __shfl_down_sync"
        ),
        sp(4),
        body(
            "The sequential parts (popping from the priority queue, checking visited, "
            "deciding which node to expand) are handled by <b>lane 0 only</b>, then "
            "broadcast to all lanes via <b>__shfl_sync</b> to keep the warp in lockstep."
        ),
        sp(),
        h2("2.3  Two Graph Variants"),
        tbl([
            ["Variant",        "Distance metric", "Vector format",  "Memory / vector", "Recall@10"],
            ["KernelGraph",    "L2 (exact)",       "128 × float32",  "512 bytes",       "0.8838"],
            ["HashKernelGraph","Hamming (approx)", "256 bits packed","32 bytes (16×↓)", "0.1963"],
        ], [3.5*cm, 3*cm, 3.5*cm, 3*cm, 2.5*cm]),
        cap("Table 1 — The Hamming variant trades recall for 16× smaller vectors and faster distance computation."),
        PageBreak(),
    ]

    # ── page 3: single query execution ────────────────────────────────────────
    story += [
        h1("3.  Single Query Execution in the GPU Kernel"),
        hr(),
        body(
            "One CUDA block (32 threads = 1 warp) handles one query end-to-end. "
            "All persistent state lives in <b>shared memory</b>; no global temporary arrays are used."
        ),
        sp(),
        h2("3.1  Shared Memory Layout per Block"),
        code(
            "┌──────────────────────────────────────────────────────────────────┐\n"
            "│                     Shared Memory (~3 KB)                        │\n"
            "│                                                                  │\n"
            "│  float sq[128]        │ Candidates<50> │ Results<50> │ bloom[256]│\n"
            "│  (query vector)       │  (min-heap)    │ (max-heap)  │ (Bloom    │\n"
            "│   512 bytes           │  ~408 bytes    │ ~408 bytes  │  filter)  │\n"
            "│                       │  BFS frontier  │ best k seen │  2 KB     │\n"
            "└──────────────────────────────────────────────────────────────────┘"
        ),
        sp(),
        h2("3.2  Execution Steps"),
        code(
            "STEP 1 — Load query (all 32 lanes, parallel)\n"
            "  Lane i loads sq[i], sq[i+32], sq[i+64], sq[i+96]   ← parallel\n"
            "  Lane 0 zeros Candidates, Results, bloom[0..255]\n"
            "  __syncthreads()\n"
            "\n"
            "STEP 2 — Seed BFS from N_MULTIPROBE=4 entry points\n"
            "  for p in 0..3:\n"
            "    ep = p * n / 4                                     ← evenly spaced\n"
            "    ALL lanes: dp = warp_l2(sq, data[ep])              ← parallel\n"
            "    Lane 0:    if not bf_test(bloom, ep):\n"
            "                   bf_set(bloom, ep)\n"
            "                   Candidates.push(dp, ep)\n"
            "                   Results.try_push(dp, ep)\n"
            "    __syncthreads()\n"
            "\n"
            "STEP 3 — Main BFS loop\n"
            "  loop:\n"
            "    Lane 0:    cn = Candidates.pop_min()               ← serial\n"
            "               __shfl_sync → broadcast cn to all lanes ← sync\n"
            "    if cn < 0: break   (frontier empty)\n"
            "\n"
            "    Lane 0:    early-stop? (cn_dist > Results.worst())\n"
            "               __shfl_sync → broadcast decision\n"
            "    if early-stop: break\n"
            "\n"
            "    for j in 0..BLOCK_SIZE-1:   (BLOCK_SIZE = 32)\n"
            "      Lane 0:  nb = adj[cn*32 + j]                     ← serial\n"
            "               __shfl_sync → broadcast nb\n"
            "      if nb < 0: continue\n"
            "\n"
            "      Lane 0:  skip = bf_test(bloom, nb)               ← serial\n"
            "               if not skip: bf_set(bloom, nb)\n"
            "               __shfl_sync → broadcast skip\n"
            "      if skip: continue\n"
            "\n"
            "      ALL lanes: nb_d = warp_l2(sq, data[nb])          ← PARALLEL ★\n"
            "\n"
            "      Lane 0:  Candidates.push(nb_d, nb)  (unconditional — bridge fix)\n"
            "               Results.try_push(nb_d, nb)\n"
            "    __syncthreads()\n"
            "\n"
            "STEP 4 — Write output\n"
            "  Lane 0:    trim Results to k, drain_best → out[qid*k..]"
        ),
        sp(4),
        body(
            "The <b>★ parallel</b> annotation marks the only line where all 32 lanes "
            "do simultaneous arithmetic. Everything else is serial in lane 0, "
            "with results broadcast via <code>__shfl_sync</code>."
        ),
        PageBreak(),
    ]

    # ── page 4: warp_l2 detail ────────────────────────────────────────────────
    story += [
        h1("4.  Warp-Level L2 Distance Computation"),
        hr(),
        body(
            "This is the hot path. For every neighbor nb encountered during BFS, "
            "all 32 lanes cooperate to compute L2(query, data[nb]) for dim=128."
        ),
        sp(),
        h2("4.1  Lane Striping"),
        code(
            "float warp_l2(const float* sq, const float* b, int dim) {\n"
            "    float sum = 0.0f;\n"
            "    for (int i = threadIdx.x; i < dim; i += 32) {  // stride-32\n"
            "        float d = sq[i] - b[i];\n"
            "        sum += d * d;\n"
            "    }\n"
            "    // butterfly reduction:\n"
            "    for (int off = 16; off >= 1; off >>= 1)\n"
            "        sum += __shfl_down_sync(0xffffffff, sum, off);\n"
            "    return sum;  // valid in lane 0\n"
            "}"
        ),
        sp(4),
        body("For dim=128, each lane handles exactly 4 elements:"),
        code(
            "Lane  0: sq[0],  sq[32], sq[64], sq[96]   → partial_sum_0\n"
            "Lane  1: sq[1],  sq[33], sq[65], sq[97]   → partial_sum_1\n"
            "  ...\n"
            "Lane 31: sq[31], sq[63], sq[95], sq[127]  → partial_sum_31"
        ),
        sp(),
        h2("4.2  Butterfly Reduction"),
        code(
            "Round 1 (off=16):  lane_i += lane_{i+16}      → 16 lanes have 2-way sums\n"
            "Round 2 (off= 8):  lane_i += lane_{i+8}       →  8 lanes have 4-way sums\n"
            "Round 3 (off= 4):  lane_i += lane_{i+4}       →  4 lanes have 8-way sums\n"
            "Round 4 (off= 2):  lane_i += lane_{i+2}       →  2 lanes have 16-way sums\n"
            "Round 5 (off= 1):  lane_0 += lane_1           → lane 0 has full L2²"
        ),
        sp(4),
        body(
            "Total: 5 rounds of <code>__shfl_down_sync</code> — a register-level "
            "operation with no shared memory traffic. The full L2 distance for a "
            "128-dim vector is computed in <b>4 FMA + 5 shuffle instructions per lane</b>, "
            "all executed simultaneously across the warp."
        ),
        sp(),
        h2("4.3  Hamming Distance Variant (warp_hamming)"),
        code(
            "int warp_hamming(const uint32_t* sq, const uint32_t* b) {\n"
            "    int lane = threadIdx.x & 31;\n"
            "    int dist = (lane < HASH_WORDS) ? __popc(sq[lane] ^ b[lane]) : 0;\n"
            "    for (int off = 16; off >= 1; off >>= 1)\n"
            "        dist += __shfl_down_sync(0xffffffff, dist, off);\n"
            "    return dist;  // valid in lane 0\n"
            "}"
        ),
        sp(4),
        body(
            "HASH_WORDS=8 (256 bits / 32). Only 8 lanes do real work; the rest contribute 0. "
            "<code>__popc</code> (population count) is a single GPU hardware instruction. "
            "Hamming distance for 256 bits = <b>8 XOR + 8 POPC + 5 shuffles</b>, "
            "far cheaper than L2's 128 multiply-adds."
        ),
        PageBreak(),
    ]

    # ── page 5: bloom filter ──────────────────────────────────────────────────
    story += [
        h1("5.  Visited Tracking: Why Bloom Filter"),
        hr(),
        body(
            "During BFS, we need to mark nodes as visited and skip them on re-encounter. "
            "On CPU this is trivial with <code>std::unordered_set</code>. "
            "On GPU inside a kernel it requires careful design."
        ),
        sp(),
        h2("5.1  Why Not the Obvious Solutions"),
        tbl([
            ["Approach",                  "Memory cost",        "Problem"],
            ["bool visited[n]\n(global)", "1 MB per block\n(n=1M)",
             "Shared memory limit is 48–96 KB. Exceeds by 10×."],
            ["bool visited[N_MQ × n]\n(global alloc)",
             "512 × 1M = 512 MB", "OOM for any realistic dataset."],
            ["std::unordered_set",        "Dynamic heap alloc",
             "No dynamic allocation inside CUDA kernels."],
            ["Bloom filter\n(shared mem)", "256 × 64 bit = 2 KB",
             "✓  Fits easily. ~1% false positive rate."],
        ], [4*cm, 3.5*cm, 8*cm]),
        cap("Table 2 — Bloom filter is the only option that fits in shared memory."),
        sp(),
        h2("5.2  Bloom Filter Implementation"),
        code(
            "// BF_WORDS=256, BF_NHASH=4  →  256×64 = 16,384 bits = 2 KB\n"
            "// 8 random 64-bit constants (same as official SONG bloomfilter.h)\n"
            "\n"
            "int bf_hash(int h, uint64_t x) {\n"
            "    x ^= x >> 33;  x *= BF_RAND[h*2];    // XOR-shift-multiply\n"
            "    x ^= x >> 33;  x *= BF_RAND[h*2+1];\n"
            "    x ^= x >> 33;\n"
            "    return (int)(x % (BF_WORDS * 64));     // position in [0, 16384)\n"
            "}\n"
            "\n"
            "void bf_set(uint64_t* bloom, int x) {\n"
            "    for (int h = 0; h < BF_NHASH; h++) {\n"
            "        int pos = bf_hash(h, x);\n"
            "        bloom[pos % BF_WORDS] |= (1ULL << (pos / BF_WORDS));\n"
            "    }\n"
            "}"
        ),
        sp(),
        h2("5.3  False Positive Rate"),
        body(
            "With BF_WORDS=256, BF_NHASH=4, and ~1,000 visited nodes per query:"
        ),
        code(
            "FPR = (1 - e^{-k·n/m})^k\n"
            "    = (1 - e^{-4 × 1000 / 16384})^4\n"
            "    ≈ 1.3 %"
        ),
        sp(4),
        body(
            "A 1.3% false positive means roughly 13 of 1,000 distinct neighbors are "
            "incorrectly skipped. This causes a small recall reduction, which is visible "
            "in our results: Recall@10 = 0.8838 vs official 0.8863 (gap = 0.0025). "
            "The gap matches what the false positive rate predicts."
        ),
        sp(),
        h2("5.4  Bit Layout (matches official SONG bloomfilter.h exactly)"),
        code(
            "word index = pos % BF_WORDS      // which uint64_t word\n"
            "bit  index = pos / BF_WORDS      // which bit within that word\n"
            "\n"
            "// Note: NOT the more common (pos / 64, pos % 64) layout.\n"
            "// This matches the official SONG source exactly."
        ),
        PageBreak(),
    ]

    # ── page 6: no STL on GPU ────────────────────────────────────────────────
    story += [
        h1("6.  Custom Implementations: Where STL Cannot Be Used"),
        hr(),
        body(
            "CUDA kernels execute on the GPU device. The C++ standard library (STL) "
            "is not available inside <code>__global__</code> functions because:"
        ),
        bul("No dynamic heap allocation (<code>malloc</code>, <code>new</code> unavailable)"),
        bul("No virtual dispatch, no exceptions"),
        bul("No OS-level threading primitives"),
        bul("Shared memory must be statically sized at compile time"),
        sp(),
        h2("6.1  What We Implemented vs What STL Provides on CPU"),
        tbl([
            ["Component",           "CPU (STL)",               "GPU kernel (custom)"],
            ["BFS frontier\n(exploration queue)",
             "std::priority_queue\n<pair<float,int>,\nvector, greater>",
             "Candidates<pq_size>\nFixed-size min-heap\nin shared memory\nO(cap) push/pop_min"],
            ["Result set\n(best-k tracker)",
             "std::priority_queue\n<pair<float,int>>",
             "Results<pq_size>\nFixed-size max-heap\nin shared memory\ntry_push / pop_worst"],
            ["Visited set",
             "std::unordered_set<int>",
             "Bloom filter\nuint64_t bloom[BF_WORDS]\nin shared memory\nbf_test / bf_set"],
            ["L2 distance",
             "for loop,\nstd::inner_product",
             "warp_l2()\nLane striping +\nbutterfly reduction\n__shfl_down_sync"],
            ["Hamming distance",
             "__builtin_popcount\n(single thread)",
             "warp_hamming()\n__popc + butterfly\nreduction across warp"],
        ], [3.5*cm, 5.5*cm, 6.5*cm]),
        cap("Table 3 — Every data structure in the GPU kernel is a custom fixed-size implementation."),
        sp(),
        h2("6.2  Candidates<pq_size> Design Detail"),
        code(
            "template<int MAX_SIZE>\n"
            "struct Candidates {          // min-heap: BFS frontier\n"
            "    float dist[MAX_SIZE];    // distances — fixed array in shared mem\n"
            "    int   id[MAX_SIZE];      // node ids\n"
            "    int   sz;               // current size\n"
            "    int   cap;              // runtime capacity (= ef_search)\n"
            "\n"
            "    // push: if full, evict the FARTHEST entry (bounds frontier size)\n"
            "    // pop_min: O(cap) linear scan — heap sort is overkill for cap<=50\n"
            "};"
        ),
        sp(4),
        body(
            "O(cap) linear scan is chosen over a proper heap because cap ≤ 50 "
            "(our maximum ef_search). For small fixed sizes, a linear scan has "
            "better cache behavior than pointer-chasing in a heap structure — "
            "all data fits in a few cache lines of shared memory."
        ),
        PageBreak(),
    ]

    # ── page 7: results ───────────────────────────────────────────────────────
    story += [
        h1("7.  Experimental Results"),
        hr(),
        body(
            "All results on SIFT1M, n=1,000,000, nq=1,000, k=10, pq=50. "
            "Ground truth computed by brute-force L2 on the same n-subset."
        ),
        sp(),
        h2("7.1  Primary Metrics (pq = 50)"),
        tbl([
            ["Method",              "Build (ms)",  "Search (ms)", "QPS",       "Recall@10"],
            ["Repro GPU Raw",       "785,934",     "8.87",        "112,736",   "0.8838"],
            ["Repro CPU Raw",       "775,734",     "312.1",       "3,205",     "0.8848"],
            ["Repro GPU Hash",      "501,106",     "10.87",       "92,024",    "0.1963"],
            ["Official CPU (wall)", "723,828",     "999.4",       "1,004",     "0.8861"],
            ["Official GPU (wall)", "842,452",     "881.8",       "1,134",     "0.8863"],
        ], [4.5*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.6*cm]),
        cap("Table 4 — Official search times are wall-clock (include disk graph loading). Repro search times are kernel-only."),
        sp(),
        h2("7.2  Recall vs ef_search Sweep"),
        tbl([
            ["ef_search (pq)", "GPU Raw recall", "CPU Raw recall", "GPU Hash recall", "Official GPU recall"],
            ["10",   "0.6277", "0.6281", "0.1712", "0.8863"],
            ["20",   "0.7519", "0.7516", "0.1854", "0.8863"],
            ["30",   "0.8154", "0.8142", "0.1915", "0.8863"],
            ["40",   "0.8549", "0.8554", "0.1941", "0.8863"],
            ["50",   "0.8838", "0.8848", "0.1963", "0.8863"],
        ], [3*cm, 3*cm, 3*cm, 3*cm, 3.5*cm]),
        cap("Table 5 — Recall@10 rises with ef_search for L2 variants. Official GPU recall is constant because\n"
            "it uses a fixed internal ef; its search time barely changes with pq."),
        sp(),
        h2("7.3  Key Observations"),
        bul(
            "<b>Recall reproduced.</b>  At pq=50, GPU Raw (0.8838) and CPU Raw (0.8848) "
            "are within 0.003 of the official (0.8863). Algorithm logic is correct."
        ),
        bul(
            "<b>GPU Hash recall is low (0.196).</b>  Expected — the 256-bit binary "
            "projection discards magnitude information. This demonstrates the accuracy / "
            "compression tradeoff the paper discusses."
        ),
        bul(
            "<b>Search timing comparison is unfair.</b>  Official 'search' includes "
            "loading the serialized graph from disk each invocation (~880 ms). "
            "Our kernel-only time is 8.87 ms. If both measured pure kernel time "
            "the gap would be much smaller."
        ),
        bul(
            "<b>Build times are comparable.</b>  All methods use single-threaded CPU "
            "NSW construction with identical parameters (EF_CONSTRUCTION=150). "
            "GPU Hash builds faster (501 s vs 786 s) because Hamming distance "
            "computation is ~16× cheaper than L2 at the same number of comparisons."
        ),
        sp(),
        h2("7.4  Total Time (Build + Upload + Search)"),
        tbl([
            ["Method",         "Build (ms)", "Upload (ms)", "Search (ms)", "Total (ms)"],
            ["Repro GPU Raw",  "785,934",    "229",         "8.87",        "786,172"],
            ["Official GPU",   "842,452",    "—",           "881.8",       "843,334"],
        ], [4.5*cm, 3*cm, 3*cm, 3*cm, 3*cm]),
        cap("Table 6 — End-to-end, Repro GPU is ~7% faster. Official 'search' still includes disk loading overhead."),
        PageBreak(),
    ]

    # ── page 8: divergences from official ───────────────────────────────────
    story += [
        h1("8.  Differences from the Official Implementation"),
        hr(),
        body(
            "The following table summarizes where the reproduction deliberately "
            "diverges from the official SONG code and why."
        ),
        sp(),
        tbl([
            ["Parameter / Design",  "Official SONG",           "This Reproduction",       "Impact"],
            ["N_MULTIPROBE",        "1 (single entry point)",  "4 (evenly spaced)",
             "Higher recall at same ef_search. Deliberate improvement."],
            ["N_MULTIQUERY",        "1 query / kernel launch", "512 queries / batch",
             "Better GPU utilization, higher QPS."],
            ["Visited tracking",    "Bloom filter\n(shared mem)",
             "Bloom filter\n(shared mem, same constants)",
             "Exact match to official."],
            ["Search timing",       "Wall time of\n`song test` process",
             "Kernel-only time\n(build / upload / search\nseparated)",
             "Not directly comparable.\nOurs excludes disk I/O."],
            ["Graph serialization", "Saves graph to disk\nafter build",
             "Kept in RAM;\nuploaded to GPU once",
             "Faster repeated queries;\nno cold-start overhead."],
            ["EF_CONSTRUCTION",     "150",                     "150",
             "Exact match."],
            ["FIXED_DEGREE",        "31",                      "31",
             "Exact match."],
            ["SEARCH_DEGREE",       "15",                      "15",
             "Exact match."],
        ], [3.5*cm, 3.5*cm, 3.5*cm, 5*cm]),
        cap("Table 7 — Parameters matching official are in rows 6–8. Deliberate improvements are rows 1–2."),
        sp(),
        h2("8.1  Bridge Node Fix (SONG Algorithm 1, Stage 3)"),
        body(
            "The official SONG paper (Algorithm 1, Stage 3) pushes every evaluated "
            "neighbor onto the BFS frontier <b>unconditionally</b>, regardless of "
            "whether it improves the result set. This ensures 'bridge nodes' — "
            "nodes that are mediocre themselves but have excellent neighbors — "
            "remain explorable."
        ),
        code(
            "// Stage 3 in our GPU kernel (kg_search_kernel):\n"
            "bool frontier_added = candidates->push(nb_d, nb);   // ALWAYS push\n"
            "bool result_added   = results->try_push(nb_d, nb);  // only if better\n"
            "\n"
            "// Without this fix: a bridge node farther than results.worst()\n"
            "// would be pruned from the frontier, cutting off its good subtree."
        ),
        sp(),
        h2("8.2  Note on Recall Gap (0.0025)"),
        body(
            "The small recall gap between Repro GPU Raw (0.8838) and Official GPU (0.8863) "
            "has two sources: (1) Bloom filter false positives (~1.3% of neighbors are "
            "incorrectly skipped), and (2) our N_MULTIPROBE=4 seeding strategy vs the "
            "official's single fixed entry point — different seeding produces slightly "
            "different BFS trajectories even at the same ef_search."
        ),
        PageBreak(),
    ]

    # ── page 9: conclusion ────────────────────────────────────────────────────
    story += [
        h1("9.  Conclusion"),
        hr(),
        body(
            "This reproduction validates the core claims of the SONG paper:"
        ),
        sp(4),
        bul(
            "<b>Algorithm correctness verified.</b>  CPU and GPU L2 variants both achieve "
            "Recall@10 ≥ 0.88 on SIFT1M, within 0.003 of the official implementation "
            "using identical graph construction parameters."
        ),
        sp(4),
        bul(
            "<b>GPU parallelism is effective for distance computation.</b>  "
            "The warp-level L2 via lane striping + butterfly reduction is the key "
            "GPU primitive. All 32 lanes execute in lockstep; no shared memory "
            "synchronization is needed for the distance step."
        ),
        sp(4),
        bul(
            "<b>Shared memory is the limiting resource.</b>  Every per-query "
            "data structure (query vector, heaps, Bloom filter) lives in shared memory "
            "to avoid global memory latency. The Bloom filter is a direct consequence "
            "of this constraint — it replaces a visited array that would be 1 MB per block."
        ),
        sp(4),
        bul(
            "<b>Hash variant demonstrates accuracy / speed tradeoff.</b>  "
            "GPU Hash builds 35% faster (501 s vs 786 s) and queries at similar "
            "throughput, but recall drops to 0.196 — the cost of discarding vector "
            "magnitude during binary projection."
        ),
        sp(4),
        bul(
            "<b>Timing methodology matters.</b>  The apparent 99× query speedup "
            "over the official implementation is an artifact of measurement: official "
            "wall time includes disk graph loading (~870 ms), while our kernel time "
            "is 8.87 ms. End-to-end (build + upload + search), our implementation "
            "is ~7% faster."
        ),
        sp(12),
        hr(),
        Paragraph(
            "Source code: <code>github.com/[repo]</code> &nbsp;·&nbsp; "
            "Dataset: SIFT1M (Jégou et al.) &nbsp;·&nbsp; "
            "Paper: Fu et al., ICDE 2020",
            ParagraphStyle("footer", parent=styles["Normal"],
                           fontSize=9, textColor=DGRAY, alignment=TA_CENTER)
        ),
    ]

    doc.build(story)
    print(f"PDF written to: {OUT}")

if __name__ == "__main__":
    build()
