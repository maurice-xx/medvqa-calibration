# P0/P1 运行报告

## 状态
warm qwen 125s
launched qwen gpu0 pid=814090 16:37:35
warm healthgpt 141s
launched healthgpt gpu1 pid=814766 16:41:26
warm gemma4 108s
launched gemma4 gpu2 pid=814938 16:47:29
qwen rc=0 17:43:39
healthgpt rc=0 17:44:44
gemma rc=1 17:44:44
rescore analysis 17:44:45
selective risk 17:45:06
aprime 17:45:20
format-cond 17:45:22

## 重评分
== qwen n=432 rho(diag,gen)=0.4183
   scoring spread (same answer, diff context) = 0.2101
   generation spread (same context, diff answers) = 0.1982
   ECE diag(g==s): {'A': 0.3847, 'B': 0.1466, 'C': 0.1694}
   ECE canonical: {'A': {'A': 0.3847, 'B': 0.2493, 'C': 0.2699}, 'B': {'A': 0.3294, 'B': 0.1466, 'C': 0.1823}, 'C': {'A': 0.3011, 'B': 0.1423, 'C': 0.1694}}
== healthgpt n=432 rho(diag,gen)=0.4339
   scoring spread (same answer, diff context) = 0.2284
   generation spread (same context, diff answers) = 0.2623
   ECE diag(g==s): {'A': 0.0892, 'B': 0.0607, 'C': 0.1131}
   ECE canonical: {'A': {'A': 0.0892, 'B': 0.1596, 'C': 0.1027}, 'B': {'A': 0.2725, 'B': 0.0607, 'C': 0.1854}, 'C': {'A': 0.1463, 'B': 0.1305, 'C': 0.1131}}
saved ./results/rescore_analysis/rescore_summary.json

## 选择性风险
      "high_conf_error_80": 0.4602,
      "high_conf_share_80": 0.9606
    },
    "threshold_from_A_at_5pct": 0.9975,
    "apply_A_thr_to_A": {
      "coverage": 0.0787,
      "risk": 0.0294
    },
    "apply_A_thr_to_B": {
      "coverage": 0.3565,
      "risk": 0.3247
    },
    "apply_A_thr_to_C": {
      "coverage": 0.3819,
      "risk": 0.2061
    }
  },
  "healthgpt500": {
    "A": {
      "risk_at_cov50": 0.0556,
      "risk_at_cov70": 0.1254,
      "risk_at_cov90": 0.1954,
      "cov_at_risk5": 0.4907,
      "cov_at_risk10": 0.6042,
      "high_conf_error_80": 0.0337,
      "high_conf_share_80": 0.412
    },
    "B": {
      "risk_at_cov50": 0.4907,
      "risk_at_cov70": 0.5446,
      "risk_at_cov90": 0.5476,
      "cov_at_risk5": 0.0023,
      "cov_at_risk10": 0.0741,
      "high_conf_error_80": 0.4955,
      "high_conf_share_80": 0.5139
    },
    "C": {
      "risk_at_cov50": 0.0972,
      "risk_at_cov70": 0.1815,
      "risk_at_cov90": 0.3008,
      "cov_at_risk5": 0.294,
      "cov_at_risk10": 0.5347,
      "high_conf_error_80": 0.113,
      "high_conf_share_80": 0.5532
    },
    "threshold_from_A_at_5pct": 0.7512,
    "apply_A_thr_to_A": {
      "coverage": 0.4907,
      "risk": 0.0472
    },
    "apply_A_thr_to_B": {
      "coverage": 0.6366,
      "risk": 0.5382
    },
    "apply_A_thr_to_C": {
      "coverage": 0.6551,
      "risk": 0.159
    }
  }
}

## A' 控制
== qwen: {"A": {"n": 432, "acc": 0.5208, "conf": 0.7834, "ece": 0.2626}, "A_prime": {"n": 432, "acc": 0.5231}, "B": {"n": 432, "acc": 0.4722, "conf": 0.9494, "ece": 0.4771}}
== healthgpt: {"A": {"n": 432, "acc": 0.7546, "conf": 0.7286, "ece": 0.0517}, "A_prime": {"n": 432, "acc": 0.706}, "B": {"n": 432, "acc": 0.4468, "conf": 0.7909, "ece": 0.3505}}
saved

## 格式条件校准
== qwen500: raw={'A': 0.2737, 'B': 0.4882, 'C': 0.4359}
   global: overall=0.0669 per={'A': 0.1388, 'B': 0.0995, 'C': 0.1196}
   cond_intercept: overall=0.0455 per={'A': 0.1001, 'B': 0.0684, 'C': 0.084}
   cond_full: overall=0.0329 per={'A': 0.0908, 'B': 0.117, 'C': 0.0506}
   independent: overall=0.0321 per={'A': 0.0908, 'B': 0.1169, 'C': 0.0517}
== healthgpt500: raw={'A': 0.0566, 'B': 0.3615, 'C': 0.1348}
   global: overall=0.0583 per={'A': 0.1216, 'B': 0.2678, 'C': 0.105}
   cond_intercept: overall=0.1036 per={'A': 0.0991, 'B': 0.1817, 'C': 0.1032}
   cond_full: overall=0.0873 per={'A': 0.0966, 'B': 0.1472, 'C': 0.0933}
   independent: overall=0.0873 per={'A': 0.0966, 'B': 0.1473, 'C': 0.0933}
== gemma500: raw={'A': 0.3636, 'B': 0.4006, 'C': 0.3826}
   global: overall=0.0734 per={'A': 0.0928, 'B': 0.0793, 'C': 0.1264}
   cond_intercept: overall=0.0732 per={'A': 0.1046, 'B': 0.0934, 'C': 0.1373}
   cond_full: overall=0.0702 per={'A': 0.0994, 'B': 0.0938, 'C': 0.0866}
   independent: overall=0.0702 per={'A': 0.0994, 'B': 0.0938, 'C': 0.0867}
saved ./results/format_cond/format_cond.json ./results/format_cond/fig6_fewshot_calibration.png
