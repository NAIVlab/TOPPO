| W&B group | arm | paper row | steps | task return | style FID |
|---|---|:---:|---:|---:|---:|
| `ampwalk_ppo_ad_cd_nnone_ln0_k1_gc1` | ppo_base | 1 | 100M | 533.56 ± 20.74 | 5.477e+04 ± 1.327e+04 |
| `ampwalk_ppo_ad_cd_npaw_lnc_k2_gc1` | ppo_popart | 2 | 100M | 534.99 ± 16.89 | 3.855e+04 ± 8706 |
| `ampwalk_ppo_ad_cfg1_npaw_lnc_k2_gc0` | ppo_fairgrad | 3 | 100M | 550.63 ± 9.01 | 3.635e+04 ± 3.943e+04 |
| `ampwalk_ppo_apc_cfg1_npaw_lnc_k2_gc0` | ppo_pcgrad | 4 | 100M | 521.11 ± 14.64 | 1.315e+04 ± 881.9 |
| `ampwalk_ppo_ad_cd_nnone_ln0_k1_gc1_reweight` | ppo_rw |  | 100M | 466.31 ± 19.07 | 2.746e+04 ± 8565 |
| `ampwalk_ppo_ad_cd_nrew_lnc_k2_gc1` | ppo_rewnorm |  | 100M | 508.10 ± 25.11 | 2.223e+04 ± 6877 |
| `ampwalk_ppo_ad_cd_nret_lnc_k2_gc1` | ppo_retscale |  | 100M | 543.65 ± 6.96 | 2.471e+04 ± 1.028e+04 |
| `ampwalk_ppo_ad_cd_nhg_lnc_k2_gc1` | ppo_hlgauss |  | 100M | 518.58 ± 15.76 | 5.587e+04 ± 1.251e+04 |
| `ampwalk_ppo_ad_cd_nhg_lnc_k2_gc1_lr1e3` | ppo_hlgauss_lr10x |  | 100M | 508.10 ± 122.96 | 3.478e+04 ± 2.751e+04 |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0` | sac_base | 5 | 100M | 573.77 ± 4.03 | 5.596e+06 ± 1.522e+06 |
| `ampwalk_sac_ad_cd_nnone_lnc_k1_gc0` | sac_ln |  | 100M | 572.70 ± 3.59 | 1.101e+07 ± 1.848e+06 |
| `ampwalk_sac_ad_cd_npaw_ln0_k1_gc0` | sac_popart |  | 100M | 574.24 ± 5.70 | 5.766e+06 ± 1.392e+06 |
| `ampwalk_sac_ad_cd_npaw_lnc_k1_gc0` | sac_ln_popart |  | 100M | 571.17 ± 4.75 | 1.238e+07 ± 2.346e+06 |
| `ampwalk_sac_ad_cd_npaw_lnc_k2_gc0` | sac_k2 |  | 100M | 564.87 ± 15.21 | 1.01e+07 ± 4.269e+06 |
| `ampwalk_sac_ad_cfg1_npaw_lnc_k2_gc0` | sac_fairgrad |  | 100M | 573.77 ± 5.76 | 4.985e+06 ± 1.048e+06 |
| `ampwalk_sac_apc_cfg1_npaw_lnc_k2_gc0` | sac_pcgrad | 6 | 100M | 575.33 ± 2.57 | 4.486e+06 ± 1.108e+06 |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_reweight` | sac_rw |  | 100M | 569.81 ± 3.41 | 3.031e+06 ± 4.048e+05 |
| `ampwalk_sac_ad_cfg1_npaw_lnc_k2_gc0_reweight` | sac_fairgrad_rw |  | 100M | 564.41 ± 3.37 | 3.228e+06 ± 7.026e+05 |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_ars` | sac_ars |  | 100M | 565.73 ± 5.40 | 3.328e+06 ± 8.708e+05 |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_arsreset` | sac_ars_reset |  | 100M | 546.87 ± 7.91 | 1.132e+07 ± 1.875e+06 |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_entmatch` | sac_targetent |  | 100M | 577.70 ± 8.99 | 7.603e+06 ± 3.165e+06 |
| `ampwalk_sac_ad_cd_nnone_ln0_k1_gc0_utd512` | sac_utd512 |  | 25M | 560.62 ± 13.64 | 6.341e+06 ± 7.357e+05 |
| `ampwalk_sac_ad_cfg1_nrew_lnc_k2_gc0` | sac_rewnorm |  | 100M | 559.29 ± 3.51 | 2.226e+06 ± 3.643e+05 |
| `ampwalk_sac_ad_cfg1_nret_lnc_k2_gc0` | sac_retscale |  | 100M | 576.14 ± 2.77 | 4.624e+06 ± 6.348e+05 |
| `ampwalk_sac_ad_cfg1_nhg_lnc_k2_gc0` | sac_hlgauss |  | 100M | 580.57 ± 1.97 | 2.376e+06 ± 1.43e+06 |

Mean ± sample std (ddof=1) over the 10 training seeds; style FID is a squared distance, lower is better. PPO and SAC FIDs differ by two orders of magnitude and are not compared across families.
