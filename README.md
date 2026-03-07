# LJ-Bench: Ontology-Based Benchmark for U.S. Crime

LJ-Bench is a benchmark for evaluating the safety of large language models (LLMs) against crime-related prompts grounded in U.S. law. It covers **76 crime types** and **630 questions**, with an augmented dataset of **13,029 questions** generated via prompt rephrasing and attack techniques.

## Repository Structure

```
├── lj_bench.csv                  # Core benchmark (630 questions, 76 crime types)
├── LJ-Bench_Augmented.csv        # Augmented dataset (13,029 questions)
├── lj-ontology.rdf / .ttl        # Crime ontology (RDF/Turtle)
├── Rephrased_questions.txt       # Rephrased variants of benchmark questions
├── Mapping_to_California_law/    # Question mappings to California Penal Code
├── Mapping_to_Canadian_Law/      # Question mappings to Canadian law
├── Mapping_to_China_Law/         # Question mappings to Chinese law
├── Mapping_to_Model_Penal_Code/  # Question mappings to MPC
├── Mapping_to_UN_Law/            # Question mappings to UN conventions
└── src/
    ├── main.py                   # Entry point for running evaluations
    ├── query_model.py            # Model querying (OpenAI, Gemini, Together AI)
    ├── format_prompt.py          # Prompt formatting and attack injection
    ├── create_augmented_dataset.py
    └── config.json               # Configuration (model, attack, API keys)
```

## Quickstart

1. **Clone the repo and install dependencies**
```bash
git clone https://github.com/your-repo/LJ-Bench.git
cd LJ-Bench
```

2. **Configure your run** in `src/config.json`:
```json
{
  "target_model": "gemini-1.5-pro",
  "target_model_type": "gemini",
  "evaluator_model": "gemini-1.5-pro",
  "input_file": "lj_bench.csv",
  "attack": "all",
  "OPENAI_API_KEY": "...",
  "GEMINI_API_KEY": "...",
  "TOGETHER_API_KEY": "..."
}
```

Supported model types: `openai`, `gemini`, `together` (for open source models).

3. **Run the attack**
```bash
python main.py --config config.json
```

4. **Run the evaluation**
```bash
python autograder.py --config eval.json
```



## Citation

```bibtex
@article{tseng2026ljbench,
  title     = {LJ-Bench: Ontology-Based Benchmark for U.S. Crime},
  author    = {Tseng, Hung Yun and Li, Wuzhen and Gkotse, Blerina and Chrysos, Grigoris},
  journal   = {Transactions on Machine Learning Research},
  year      = {2026}
}
```
