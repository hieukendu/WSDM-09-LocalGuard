# Secondary aggregate results

Exploratory, union-auditor-relative; no human labels. 300 matched user-level splits per task. These are split-averaged held-out means, not confidence bounds.

## Uniform and weighted losses

| Task | alpha | Loss | Coarse loss / admitted | Fine loss / admitted |
|---|---:|---|---:|---:|
| lampqa | 0.2 | count | 0.107 / 2.177 | 0.161 / 2.593 |
| lampqa | 0.2 | severity2 | 0.086 / 1.280 | 0.136 / 2.175 |
| lampqa | 0.5 | count | 0.385 / 4.029 | 0.421 / 4.137 |
| lampqa | 0.5 | severity2 | 0.351 / 3.548 | 0.408 / 3.794 |
| lamp3 | 0.2 | count | 0.150 / 3.109 | 0.172 / 3.334 |
| lamp3 | 0.2 | severity2 | 0.090 / 1.982 | 0.151 / 2.775 |
| lamp3 | 0.5 | count | 0.405 / 4.480 | 0.425 / 4.525 |
| lamp3 | 0.5 | severity2 | 0.349 / 4.039 | 0.420 / 4.347 |
| lamp7 | 0.2 | count | 0.000 / 0.000 | 0.150 / 1.530 |
| lamp7 | 0.2 | severity2 | 0.000 / 0.000 | 0.124 / 1.142 |
| lamp7 | 0.5 | count | 0.297 / 2.328 | 0.443 / 2.849 |
| lamp7 | 0.5 | severity2 | 0.348 / 2.282 | 0.411 / 2.508 |
| lamp2 | 0.2 | count | 0.087 / 2.771 | 0.134 / 3.331 |
| lamp2 | 0.2 | severity2 | 0.042 / 0.760 | 0.066 / 2.031 |
| lamp2 | 0.5 | count | 0.294 / 4.549 | 0.360 / 4.804 |
| lamp2 | 0.5 | severity2 | 0.268 / 4.453 | 0.340 / 4.643 |

## Training-fixed bins and finer-grid coverage

| Task | Train cut | Short / long users | Within-bin calibration n range | Grid sizes | Missing reader calls |
|---|---:|---:|---:|---:|---|
| lampqa | 32.0 | 165 / 135 | 55–95 | 8 to 45 | llama31_8b: 823; qwen25_7b: 385 |
| lamp3 | 142.5 | 238 / 212 | 90–135 | 8 to 44 | llama31_8b: 311; qwen25_14b: 678; qwen25_7b: 951 |
| lamp7 | 14.0 | 167 / 133 | 57–93 | 8 to 45 | qwen25_7b: 647 |
| lamp2 | 12.5 | 71 / 79 | 26–49 | 8 to 44 | no completed full reader cell |

## Cumulative unique disclosures: synthetic arrivals/rankings on real profiles

No observed multi-query histories, new retrieval, or reader utility. Fixed top-8 pool only.

| Task | alpha | T | Single-prompt calibration: unique records / categories | Cumulative calibration: unique records / categories |
|---|---:|---:|---:|---:|
| lampqa | 0.2 | 1 | 0.107 / 0.144 | 0.107 / 0.144 |
| lampqa | 0.2 | 2 | 0.107 / 0.144 | 0.106 / 0.143 |
| lampqa | 0.2 | 5 | 0.107 / 0.144 | 0.106 / 0.143 |
| lampqa | 0.2 | 10 | 0.107 / 0.144 | 0.106 / 0.143 |
| lampqa | 0.5 | 1 | 0.385 / 0.540 | 0.385 / 0.540 |
| lampqa | 0.5 | 2 | 0.419 / 0.579 | 0.372 / 0.498 |
| lampqa | 0.5 | 5 | 0.441 / 0.604 | 0.381 / 0.502 |
| lampqa | 0.5 | 10 | 0.441 / 0.604 | 0.381 / 0.502 |
| lamp3 | 0.2 | 1 | 0.136 / 0.185 | 0.136 / 0.185 |
| lamp3 | 0.2 | 2 | 0.144 / 0.195 | 0.123 / 0.166 |
| lamp3 | 0.2 | 5 | 0.150 / 0.202 | 0.124 / 0.166 |
| lamp3 | 0.2 | 10 | 0.150 / 0.202 | 0.124 / 0.166 |
| lamp3 | 0.5 | 1 | 0.375 / 0.495 | 0.375 / 0.495 |
| lamp3 | 0.5 | 2 | 0.455 / 0.574 | 0.367 / 0.481 |
| lamp3 | 0.5 | 5 | 0.490 / 0.616 | 0.351 / 0.466 |
| lamp3 | 0.5 | 10 | 0.490 / 0.616 | 0.351 / 0.466 |
| lamp7 | 0.2 | 1 | 0.000 / 0.000 | 0.000 / 0.000 |
| lamp7 | 0.2 | 2 | 0.000 / 0.000 | 0.000 / 0.000 |
| lamp7 | 0.2 | 5 | 0.000 / 0.000 | 0.000 / 0.000 |
| lamp7 | 0.2 | 10 | 0.000 / 0.000 | 0.000 / 0.000 |
| lamp7 | 0.5 | 1 | 0.301 / 0.328 | 0.301 / 0.328 |
| lamp7 | 0.5 | 2 | 0.301 / 0.328 | 0.301 / 0.328 |
| lamp7 | 0.5 | 5 | 0.301 / 0.328 | 0.301 / 0.328 |
| lamp7 | 0.5 | 10 | 0.301 / 0.328 | 0.301 / 0.328 |
| lamp2 | 0.2 | 1 | 0.078 / 0.145 | 0.078 / 0.145 |
| lamp2 | 0.2 | 2 | 0.078 / 0.145 | 0.066 / 0.125 |
| lamp2 | 0.2 | 5 | 0.078 / 0.145 | 0.066 / 0.125 |
| lamp2 | 0.2 | 10 | 0.078 / 0.145 | 0.066 / 0.125 |
| lamp2 | 0.5 | 1 | 0.287 / 0.484 | 0.287 / 0.484 |
| lamp2 | 0.5 | 2 | 0.338 / 0.569 | 0.260 / 0.420 |
| lamp2 | 0.5 | 5 | 0.351 / 0.590 | 0.259 / 0.418 |
| lamp2 | 0.5 | 10 | 0.351 / 0.590 | 0.259 / 0.418 |

## LaMP-2 pilot (not independent confirmation)

n=60; accuracy no-profile=0.333, top-5=0.517; paired gain=+0.183, 95% bootstrap interval [+0.067, +0.300].
