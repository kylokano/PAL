# PAL: Persona-Aware Alignment Framework for Personalized Dialogue

GenerationPersona-Aware Alignment Framework

This repository contains the implementation of PAL (Persona-Aware Alignment Framework for Personalized Dialogue
GenerationPersona-Aware Alignment Framework), a novel approach for persona alignment in dialogue systems. PAL treats persona alignment as the primary optimization objective and employs a two-stage training approach with a "Select then Generate" inference strategy to improve persona sensitivity and generate persona-relevant responses.

## Data Structure and Paths

### Required Data Paths

To run PAL effectively, configure these path parameters in your configuration files (`config/config_dialogue.yaml`):

#### Language-Specific Data Directories

- **`chinese_data_dir`**: Path to Chinese persona dialogue dataset
  
  - Default: `"./data"` (if not specified)
  - Contains: `/path/to/chinese/data/train_ch.txt`, `/path/to/chinese/data/valid#1_ch.txt`, etc.
- **`english_data_dir`**: Path to English persona dialogue dataset
  
  - Default: `"./data"` (if not specified)
  - Contains: `/path/to/english/data/train_origin.txt`, `/path/to/english/data/valid_origin.txt`, etc.

### Dataset Format

Both Chinese and English datasets follow the same JSON structure:

```json
[
  [
    ["persona1", "persona2", "persona3"],  // List of personas for this dialogue
    ["user1_first_turn", "user2_response"], // Dialogue history
    "user1_next_response",                 // Target response
    [2]                       // Persona labels (2 = relevant persona index)
  ]
]
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

Please fill the path in scripts, then run

```bash
# GPT2
bash scripts/gpt2.sh

# Llama3
bash scripts/llama3.s
```

### Key Training Parameters

- `data_mode`: `"select"`, `"generate"`, `"mixture"`, `"generate_for_dpo"`, `"predict_select_then_generate"`
- `is_chinese`: Boolean flag for Chinese vs English dataset
- `only_persona_response`: Whether to only return persona-relevant responses

## Custom Data Paths

To use custom data paths, update your configuration file or override via command line:

```yaml
# Add to your config file
chinese_data_dir: "/your/custom/chinese/data/path"
english_data_dir: "/your/custom/english/data/path"
chinese_select_data_path: "/your/custom/chinese/select/results.json"
english_select_data_path: "/your/custom/english/select/results.json"
```

Or override via CLI:

```bash
python train.py chinese_data_dir=/custom/chinese/path english_data_dir=/custom/english/path
```

