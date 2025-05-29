SYSTEM_PROMPT_1 = (
    
    """
    looking at the json guideline, json example, and json dictionary,
    please generate an example for:
    """
)

SYSTEM_PROMPT_2 = (
    """
    follow the similar pipeline as the example, 
    but you can choose different agents if they function the same.

    Generate a configuration structure in **JSON format** based on the following guidelines and examples. The output should be a single JSON object, starting with a top-level key "chunks".
    Let's make sure we have the reader at the beginning of the pipeline.
    
[JSON Structure Guidelines for Pipeline Configuration]

1.  **Hierarchy:** The JSON object has a top-level key `"chunks"`. The value of `"chunks"` is an object containing one or more *supernode* objects. Each *supernode* object is keyed by its unique `supernode_name` (string).
    * Each *supernode* object contains one or more *chunk* objects. Each *chunk* object is keyed by its `chunk_name` (string, unique within the supernode).
    * Each *chunk* object contains key-value pairs for its inputs (see Rule 4) and an `"agents"` object.
    * The `"agents"` object contains one or more *agent* objects. Each *agent* object is keyed by its `agent_name` (see Rule 5 & 7).
    * Each *agent* object contains key-value pairs representing its specific parameters (see Rule 5) and potentially its input sources (see Rule 6) or special flags (`"supernode_output"`, `"chunk_output"`).

    {
      "chunks": {
        "supernode_name_1": {
          "chunk_name_1": {
            // Chunk Inputs here... (Rule 4)
            "agents": {
              "agent_name_inventory_match_1": { /* Agent parameters verified by inventory (Rule 5) */ },
              "agent_name_inventory_match_2": { /* Agent parameters verified by inventory (Rule 5) */ }
            }
          },
          "chunk_name_2": {
            // Chunk Inputs here... (Rule 4)
            "agents": {
              "agent_name_inventory_match_1": { /* Agent parameters verified by inventory (Rule 5) */ }
            }
          }
        },
        "supernode_name_2": {
          // ... structure continues
        }
      }
    }

2.  **Supernode Output:** Within the `"agents"` object for *all* chunks belonging to a single supernode, exactly *one* agent object *must* contain the key-value pair `"supernode_output": true`. This designates that agent's primary result (as defined in its `agent_output_def` in the inventory) as the final output for the entire supernode.

3.  **Chunk Output:** An agent object can *optionally* contain the key-value pair `"chunk_output": true`. If present for an agent other than the last one defined in the chunk, this agent's primary output becomes the accessible output of that chunk for other chunks within the *same* supernode. If no agent has `"chunk_output": true`, the primary output of the *last* agent defined in the chunk is implicitly the chunk's output.

4.  **Chunk Inputs:**
    * Inputs for a chunk are defined as key-value pairs directly within the *chunk* object (as siblings to the `"agents"` object).
    * Keys typically follow the pattern `"input"` or `"input_N"` (e.g., `"input_1"`, `"input_2"`). These serve as aliases that can be referenced by agents within the chunk (see Rule 6). The specific `input_N` names should correspond logically to the required `agent_input_def` names found within the chunk's agents (e.g., if an agent needs `input_1` and `input_2`, the chunk should define sources for these).
    * The value (string) specifies the source:
        * From another supernode: `"supernode_name"` (e.g., `"input": "chest_xr_image"`)
        * From another chunk within the same supernode: `"from chunk_name"` (e.g., `"input_2": "from image_processing"`)
        * From a specific named output of another chunk: `"from chunk_name output_name"` (e.g., `"input_2": "from neural_net mask"`). The `output_name` must match a key in the `agent_output_def` section of the source agent's inventory definition.
    * To explicitly provide no input for an *optional* agent input (as defined in the inventory), either omit the corresponding chunk input or omit the agent-level input mapping (see Rule 6). If an agent requires a specific value like `"off"` to disable functionality via an *optional* input, define it at the chunk level: `"input_3": "off"`.

    // Example Chunk with Inputs
    "some_chunk": {
      "input_1": "source_supernode_name", // Alias for input_1 source
      "input_2": "from previous_chunk_in_same_supernode", // Alias for input_2 source
      "input_3": "off", // Alias for input_3, potentially disabling optional functionality
      "agents": {
        // ... agents definition referencing "input_1", "input_2", "input_3" (Rule 6)
      }
    }

5.  **Agents and Agent Parameters:**
    * Each key under the `"agents"` object is the `agent_name`. This name **must** exactly match a key in the `verifier inventory` (e.g., `bounding_box`, `tf2_segmentation`, `resize`), including case (typically lowercase with underscores). See Rule 7 for uniqueness and suffixes.
    * The value associated with the `agent_name` key is an object containing the agent's parameters.
    * Parameter keys **must** exactly match the names defined in the `agent_parameter_def` section of the inventory for that specific agent (e.g., `"max_hole_size"`, `"is_3d"`, `"logical_operator"`), including case.
    * Parameter values **must** use standard JSON types corresponding to the `type` specified in the inventory:
        * Strings: Enclosed in double quotes (`"example_string"`, `"and"`, `"simplemind/path/settings.yaml"`).
        * Numbers: Integers (`5000`) or floating-point (`0.5`, `0.03`). Not enclosed in quotes.
        * Booleans: `true` or `false`. Not enclosed in quotes.
        * Arrays: Use JSON arrays `[]` for lists/tuples specified in the inventory (e.g., `[512, 512]`, `["item1", "item2"]`).
        * Objects: Use JSON objects `{}` for nested structures defined in the inventory.
    * Only include parameters required by the specific workflow. Optional parameters (marked `optional: true` in the inventory) can be omitted to use their default values (also specified in the inventory). Required parameters (`optional: false`) **must** be included.

6.  **Agent Inputs (Overrides):**
    * By default, agents implicitly consume chunk inputs based on matching names (e.g., an agent needing `input_1` will use the chunk's `"input_1"` source).
    * To explicitly map sources to an agent's inputs (defined in its `agent_input_def` in the inventory), define them within the agent's parameter object using keys matching the required input names (e.g., `"input_1"`, `"input_2"`, or alternate names like `"image"`, `"mask"` if specified in the inventory).
    * The value (string) specifies the source:
        * From a chunk input alias: `"input_N"` (e.g., `"input_1": "input_1"` explicitly uses the source defined by the chunk's `"input_1"` key).
        * From a previous agent within the *same* chunk: `"previous_agent_name"` (e.g., `"input_1": "resize"` refers to the primary output of the agent keyed as `resize` in the same chunk).
        * From a specific named output of a previous agent: `"previous_agent_name output_name"` (e.g., `"input_1": "tf2_segmentation mask"` refers to the `mask` output of the `tf2_segmentation` agent in the same chunk). The `output_name` must match a key in the `agent_output_def` of the source agent's inventory definition.

    "spatial_inference": { // Example Chunk Name
      "input_1": "ngt_safe_zone_chest_xr", // Chunk input alias 1 (source: supernode)
      "input_2": "ngt_chest_xr", // Chunk input alias 2 (source: supernode)
      "agents": {
        "mask_logic": { // Agent 1 (matches mask_logic in inventory)
          "input_1": "input_1", // Explicitly maps chunk's alias "input_1" to this agent's input_1
          "logical_operator": "not" // Parameter verified by inventory
          // input_2 is optional per inventory, not provided, agent uses default behavior (e.g., ignores it)
          // none_if_empty is optional per inventory, defaults to false
        },
        "mask_logic-1": { // Agent 2 (suffix needed for uniqueness, Rule 7)
          "input_1": "mask_logic", // Takes default output (likely 'mask') from agent 'mask_logic'
          "input_2": "input_2", // Takes chunk's alias "input_2"
          "logical_operator": "and", // Parameter verified by inventory
          "supernode_output": true // Control Flag
        }
      }
    }

7.  **Naming Requirements and Conventions:**
    * `supernode_name`: Must be unique across the entire JSON. Convention: `object_modality` (e.g., `"trachea_chest_xr"`, `"kidney_abd_ct"`).
    * `chunk_name`: Must be unique *within* a supernode. Standard names (`"load_image"`, `"image_processing"`, `"neural_net"`, `"mask_processing"`, `"spatial_inference"`, `"candidate_select"`, `"save"`) are preferred. Use suffixes for variations within a supernode (e.g., `"neural_net_cnn"`, `"mask_processing_cleanup"`).
    * `agent_name`: **Must** match a key in the `verifier inventory` exactly (case-sensitive). Must be unique *within* a chunk. If the *same* agent script needs to be run multiple times within a single chunk, append a suffix like `-1`, `-2`, etc., to the inventory key (e.g., `"mask_logic"`, `"mask_logic-1"`).


[json example]
{
  "chunks": {
    "chest_xr_image": {
      "load_image": {
        "agents": {
          "reader": {
            "csv_path": "__input_images__",
            "header_params": "__header_params__",
            "supernode_output": true
          },
          "save_image": {
            "output_filename": "chest_xr_input_image"
          }
        }
      }
    },
    "trachea_chest_xr": {
      "image_processing": {
        "input": "chest_xr_image",
        "agents": {
          "resize": {
            "target_shape": "[512, 512]",
            "order": 3,
            "preserve_range": true,
            "anti_aliasing": true,
            "numpy_only": true
          },
          "expand_channels": {
            "number_of_channels": 1,
            "numpy_only": true
          },
          "clahe": {
            "nbins": 256,
            "clip_limit": 0.03,
            "channel": 0,
            "numpy_only": true
          },
          "z_score": {
            "channel": 0,
            "numpy_only": true
          }
        }
      },
      "save_ip": {
        "input": "from image_processing",
        "agents": {
          "save_image": {
            "output_filename": "chest_xr_trachea_preprocessing"
          }
        }
      },
      "neural_net": {
        "input_1": "chest_xr_image",
        "input_2": "from image_processing",
        "agents": {
          "tf2_segmentation": {
            "prediction_threshold": 0.5,
            "settings_yaml": "simplemind/example/sub/cnn_settings/trachea_cnn_settings.yaml",
            "working_dir": "trachea_chest_xr",
            "weights_path": "simplemind/example/sub/weights/trachea_chest_xr",
            "weights_url": "https://drive.google.com/file/d/14JN65KIwNCYb_Lq50p3LMKXlaS4lnyR1/view?usp=sharing"
          }
        }
      },
      "save_cnn": {
        "input_1": "chest_xr_image",
        "input_2": "from neural_net",
        "agents": {
          "save_image": {
            "output_filename": "chest_xr_trachea_cnn_mask"
          }
        }
      },
      "mask_processing": {
        "input": "from neural_net",
        "agents": {
          "morphology": {
            "morphological_task": "open",
            "kernel": "['ellipse', 15, 15]"
          }
        }
      },
      "save_mp": {
        "input_1": "chest_xr_image",
        "input_2": "from mask_processing",
        "agents": {
          "save_image": {
            "output_filename": "chest_xr_trachea_open_mask"
          }
        }
      },
      "candidate_select": {
        "input": "from mask_processing",
        "agents": {
          "connected_components": {
            "connectivity": 6,
            "voxel_count_threshold": 5000
          },
          "decision_tree": {
            "input_1": "agent_1",
            "DT_dict_path": "simplemind/example/sub/DT_yaml/trachea_dt.yaml",
            "visualize_png": true
          },
          "candidate_selector": {
            "input_1": "agent_2",
            "input_2": "agent_1",
            "confidence_thres": 1,
            "largest_only": true,
            "supernode_output": true
          }
        }
      },
      "save": {
        "input_1": "chest_xr_image",
        "input_2": "trachea_chest_xr",
        "agents": {
          "save_image": {
            "mask_alpha": 0.5,
            "output_filename": "chest_xr_trachea"
          }
        }
      }
    },
    "lungs_chest_xr": {
      "image_processing": {
        "input": "chest_xr_image",
        "agents": {
          "resize": {
            "target_shape": "[512, 512]",
            "order": 3,
            "preserve_range": true,
            "anti_aliasing": true,
            "numpy_only": true
          },
          "expand_channels": {
            "number_of_channels": 1,
            "numpy_only": true
          },
          "clahe": {
            "nbins": 256,
            "clip_limit": 0.03,
            "channel": 0,
            "numpy_only": true,
            "chunk_output": true
          },
          "save_image": {
            "input_1": "chest_xr_image",
            "output_filename": "chest_xr_lungs_preprocessing"
          }
        }
      },
      "neural_net": {
        "input_1": "chest_xr_image",
        "input_2": "from image_processing",
        "agents": {
          "tf2_segmentation": {
            "prediction_threshold": 0.5,
            "settings_yaml": "simplemind/example/sub/cnn_settings/lungs_cnn_settings.yaml",
            "working_dir": "lungs_chest_xr",
            "weights_path": "simplemind/example/sub/weights/lungs_chest_xr",
            "weights_url": "https://drive.google.com/file/d/1pjNu3fzThjPlw6wHne2OpThKrQugPlTx/view?usp=sharing",
            "supernode_output": true
          }
        }
      },
      "save": {
        "input_1": "chest_xr_image",
        "input_2": "lungs_chest_xr",
        "agents": {
          "save_image": {
            "mask_alpha": 0.5,
            "output_filename": "chest_xr_lungs"
          }
        }
      }
    }
  }
}


[json dictionary]
{
  "bounding_box": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "mask"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "mask": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "axis": {
          "default": "z",
          "optional": true
        },
        "offset_unit": {
          "default": "NA",
          "optional": true
        },
        "slice_wise_bounding_box": {
          "default": false,
          "optional": true
        },
        "x_lower_offset": {
          "default": 0,
          "optional": true
        },
        "x_upper_offset": {
          "default": 0,
          "optional": true
        },
        "y_lower_offset": {
          "default": 0,
          "optional": true
        },
        "y_upper_offset": {
          "default": 0,
          "optional": true
        },
        "z_lower_offset": {
          "default": 0,
          "optional": true
        },
        "z_upper_offset": {
          "default": 0,
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/mask_processing/bounding_box.py"
  },
  "candidate_selector": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "alternate_names": [
            "decision_tree_data"
          ],
          "optional": false,
          "type": "dictionary"
        },
        "input_2": {
          "alternate_names": [
            "candidate_data"
          ],
          "optional": true,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "mask": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "accept_blank_image": {
          "default": false,
          "optional": true
        },
        "confidence_thres": {
          "optional": false
        },
        "largest_only": {
          "default": false,
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/reasoning/candidate_selector.py"
  },
  "clahe": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "channel": {
          "optional": true
        },
        "clip_limit": {
          "optional": false
        },
        "nbins": {
          "optional": false
        },
        "numpy_only": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/image_processing/clahe.py"
  },
  "connected_components": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "mask"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "connectivity": {
          "optional": false
        },
        "voxel_count_threshold": {
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/mask_processing/connected_components.py"
  },
  "crop": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        },
        "input_2": {
          "alternate_names": [
            "bounding_box"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {}
    },
    "path": "simplemind/agent/image_processing/crop.py"
  },
  "decision_tree": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "alternate_names": [
            "candidate_data"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_2": {
          "alternate_names": [
            "reference_data"
          ],
          "optional": true,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "DT_dict": {
          "type": "dictionary"
        }
      },
      "agent_parameter_def": {
        "DT_dict_path": {
          "generate_params": "<function Decisiontree.generate_dt_params at 0x708caf337e20>",
          "optional": false
        },
        "learn": {
          "default": false,
          "optional": true
        },
        "learn_output_name": {
          "default": "dt_train",
          "optional": true
        },
        "ref_iou_threshold": {
          "default": 0.7,
          "optional": true
        },
        "visualize_png": {
          "default": false,
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/reasoning/decision_tree.py"
  },
  "edges": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "alternate_names": [
            "mask_1"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "mask": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {}
    },
    "path": "simplemind/agent/mask_processing/edges.py"
  },
  "eval_mask": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "optional": false,
          "type": "image_compressed_numpy"
        },
        "input_2": {
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_3": {
          "optional": false,
          "type": "string"
        }
      },
      "agent_output_def": {
        "metrics": {
          "type": "dictionary"
        }
      },
      "agent_parameter_def": {}
    },
    "path": "simplemind/agent/eval/eval_mask.py"
  },
  "expand_channels": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "number_of_channels": {
          "optional": false
        },
        "numpy_only": {
          "default": false,
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/image_processing/expand_channels.py"
  },
  "export_nifti": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "alternate_names": [
            "input",
            "image"
          ],
          "type": "image_compressed_numpy"
        },
        "input_2": {
          "alternate_names": [
            "mask"
          ],
          "optional": true,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {},
      "agent_parameter_def": {
        "csv_filename": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/image_processing/export_nifti.py"
  },
  "get_image_mask": {
    "info": {
      "agent_input_def": {
        "input": {
          "optional": false,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "mask": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "x_lower_prop": {
          "default": 0,
          "optional": true
        },
        "x_upper_prop": {
          "default": 0,
          "optional": true
        },
        "y_lower_prop": {
          "default": 0,
          "optional": true
        },
        "y_upper_prop": {
          "default": 0,
          "optional": true
        },
        "z_lower_prop": {
          "default": 0,
          "optional": true
        },
        "z_upper_prop": {
          "default": 0,
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/mask_processing/get_image_mask.py"
  },
  "histeq": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "channel": {
          "optional": false
        },
        "nbins": {
          "optional": false
        },
        "numpy_only": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/image_processing/histeq.py"
  },
  "hole_filling": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "mask"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "mask": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "is_3d": {
          "optional": false
        },
        "max_hole_size": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/mask_processing/hole_filling.py"
  },
  "mask_logic": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "alternate_names": [
            "mask_1"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_2": {
          "alternate_names": [
            "mask_2"
          ],
          "default": false,
          "optional": true,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "mask": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "logical_operator": {
          "optional": false
        },
        "none_if_empty": {
          "default": false,
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/mask_processing/mask_logic.py"
  },
  "measure_distance": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_2": {
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_3": {
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "distance": {
          "type": "metric"
        }
      },
      "agent_parameter_def": {}
    },
    "path": "simplemind/agent/eval/dev/measure_distance.py"
  },
  "min_max": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "channel": {
          "optional": true
        },
        "numpy_only": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/image_processing/min_max.py"
  },
  "morphology": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "mask"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "mask": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "kernel": {
          "generate_params": "<function Morphologicalprocessingagent.generate_kernel_params at 0x708b79595630>",
          "optional": false
        },
        "morphological_task": {
          "generate_params": "<function Morphologicalprocessingagent.generate_task_param at 0x708b795955a0>",
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/mask_processing/morphology.py"
  },
  "mr_prostate_segmentation_inference": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "gpu_num": {
          "optional": true
        },
        "model_backbone": {
          "optional": false
        },
        "model_num_classes": {
          "optional": false
        },
        "model_output_class": {
          "default": 0,
          "optional": true
        },
        "model_weight_dir": {
          "optional": true
        },
        "output_class_select": {
          "default": false,
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/nn/torch/task_specific/mr_prostate_segmentation_inference.py"
  },
  "mr_prostate_segmentation_preprocessor": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {}
    },
    "path": "simplemind/agent/image_processing/task_specific/mr_prostate_segmentation_preprocessor.py"
  },
  "nn_preprocessing": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "numpy_only": {
          "default": false,
          "optional": true
        },
        "settings_yaml": {
          "optional": false
        },
        "training_label": {
          "default": null,
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/image_processing/nn_preprocessing.py"
  },
  "old_morphology": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "mask"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "mask": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "kernel_shape": {
          "optional": false
        },
        "kernel_size": {
          "optional": false
        },
        "morphological_task": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/mask_processing/old_morphology.py"
  },
  "plot_cxr": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        },
        "input_2": {
          "alternate_names": [
            "et_tip"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_3": {
          "alternate_names": [
            "carina"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_4": {
          "alternate_names": [
            "et_tube"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_5": {
          "alternate_names": [
            "ett_alert"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_6": {
          "alternate_names": [
            "et_zone"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_7": {
          "alternate_names": [
            "distance"
          ],
          "optional": false,
          "type": "metric"
        }
      },
      "agent_output_def": {},
      "agent_parameter_def": {
        "mask_alpha": {
          "default": 0.8,
          "optional": true
        },
        "output_filename": {
          "default": "cxr_final_result",
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/image_processing/dev/plot_cxr.py"
  },
  "pyradiomics_agent": {
    "info": {
      "agent_input_def": {
        "input1": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        },
        "input2": {
          "alternate_names": [
            "roi"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "feature_dict": {
          "type": "dictionary"
        }
      },
      "agent_parameter_def": {
        "crop": {
          "default": true,
          "optional": true
        },
        "roi_index": {
          "default": 1,
          "optional": true
        },
        "settings_file": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/feature_extraction/pyradiomics_agent.py"
  },
  "reasoning": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "alternate_names": [
            "candidate_data"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        },
        "input_2": {
          "alternate_names": [
            "reference_data"
          ],
          "default": null,
          "optional": true,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "DT_dict": {
          "type": "dictionary"
        }
      },
      "agent_parameter_def": {
        "confidence_class": {
          "default": "FuzzyConfidence",
          "optional": true
        },
        "reasoning_settings_yaml": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/reasoning/reasoning.py"
  },
  "resize": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "anti_aliasing": {
          "optional": false
        },
        "numpy_only": {
          "default": false,
          "optional": true
        },
        "order": {
          "optional": false
        },
        "preserve_range": {
          "optional": false
        },
        "target_shape": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/image_processing/resize.py"
  },
  "save_image": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "alternate_names": [
            "input",
            "image"
          ],
          "optional": true,
          "type": "image_compressed_numpy"
        },
        "input_2": {
          "alternate_names": [
            "mask"
          ],
          "optional": true,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {},
      "agent_parameter_def": {
        "mask_alpha": {
          "default": 0.8,
          "optional": true
        },
        "mask_color": {
          "optional": true
        },
        "not_none_input": {
          "default": false,
          "optional": true
        },
        "output_filename": {
          "optional": false
        },
        "title": {
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/image_processing/save_image.py"
  },
  "set_metadata": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "new_metadata": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/image_processing/set_metadata.py"
  },
  "spatial_offset": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "mask"
          ],
          "optional": false,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "x_offset_1": {
          "optional": true
        },
        "x_offset_2": {
          "optional": true
        },
        "y_offset_1": {
          "optional": true
        },
        "y_offset_2": {
          "optional": true
        },
        "z_offset_1": {
          "optional": true
        },
        "z_offset_2": {
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/mask_processing/spatial_offset.py"
  },
  "tf2_segmentation": {
    "info": {
      "agent_input_def": {
        "input_1": {
          "alternate_names": [
            "original_image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        },
        "input_2": {
          "alternate_names": [
            "preprocessed_image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        },
        "input_3": {
          "alternate_names": [
            "bounding_box"
          ],
          "default": null,
          "optional": true,
          "type": "mask_compressed_numpy"
        }
      },
      "agent_output_def": {
        "mask": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "learn": {
          "default": false,
          "optional": true
        },
        "output_map": {
          "default": false,
          "optional": true
        },
        "prediction_threshold": {
          "default": 0.5,
          "optional": true
        },
        "settings_yaml": {
          "optional": false
        },
        "use_checkpoint": {
          "default": false,
          "optional": true
        },
        "weights_path": {
          "default": null,
          "optional": true
        },
        "working_dir": {
          "default": null,
          "optional": true
        }
      }
    },
    "path": "simplemind/agent/nn/tf2/tf2_segmentation.py"
  },
  "threshold_hu": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "mask": {
          "type": "mask_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "lower_hu_limit": {
          "optional": false
        },
        "upper_hu_limit": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/image_processing/threshold_hu.py"
  },
  "window_level": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "channel": {
          "optional": false
        },
        "level": {
          "optional": false
        },
        "numpy_only": {
          "default": false,
          "optional": true
        },
        "window": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/image_processing/window_level.py"
  },
  "z_score": {
    "info": {
      "agent_input_def": {
        "input": {
          "alternate_names": [
            "image"
          ],
          "optional": false,
          "type": "image_compressed_numpy"
        }
      },
      "agent_output_def": {
        "image": {
          "type": "image_compressed_numpy"
        }
      },
      "agent_parameter_def": {
        "channel": {
          "optional": false
        },
        "numpy_only": {
          "optional": false
        }
      }
    },
    "path": "simplemind/agent/image_processing/z_score.py"
  }
}
    """
)