## Render: 120 scenarios, 2048x1280, 20 frames x 4 rounds per variant

| Variant | Scenarios | GPU time vs shipped (geo mean) | Best | Worst | Faster | Same | Slower | Max pixel diff |
|---|---|---|---|---|---|---|---|---|
| ea_exp | 36 | 1.045 | 0.964 | 1.126 | 0 | 33 | 3 | 0.295 |
| lab_first | 72 | 0.883 | 0.489 | 1.088 | 22 | 50 | 0 | 0.000 |
| clip | 72 | 0.872 | 0.551 | 1.100 | 26 | 44 | 2 | 1.000 |
| skip4 | 120 | 0.857 | 0.272 | 1.918 | 51 | 46 | 23 | 1.000 |
| skip8 | 120 | 0.814 | 0.170 | 1.651 | 53 | 43 | 24 | 1.000 |
| skip16 | 120 | 0.833 | 0.125 | 1.583 | 54 | 33 | 33 | 1.000 |
| lut | 84 | 1.011 | 0.898 | 1.114 | 0 | 83 | 1 | 0.000 |
| idbuf | 84 | 0.996 | 0.870 | 1.074 | 0 | 84 | 0 | 0.000 |
| override | 30 | 0.869 | 0.723 | 1.018 | 16 | 14 | 0 | 0.000 |

### ea_exp

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.34 | 1.104 | slower | 3.749% |
| dnaA_xy1 home | imglab | EA | 1.17 | 1.19 | 1.021 | same | 2.129% |
| dnaA_xy1 home | imglab50 | EA | 1.17 | 1.20 | 1.023 | same | 2.956% |
| dnaA_xy1 zoom | img | EA | 1.19 | 1.33 | 1.119 | slower | 11.794% |
| dnaA_xy1 zoom | imglab | EA | 5.97 | 6.11 | 1.023 | same | 2.157% |
| dnaA_xy1 zoom | imglab50 | EA | 7.02 | 7.48 | 1.065 | same | 6.934% |
| dnaA_xy1 top | img | EA | 0.55 | 0.61 | 1.110 | same | 10.829% |
| dnaA_xy1 top | imglab | EA | 1.85 | 1.91 | 1.036 | same | 4.454% |
| dnaA_xy1 top | imglab50 | EA | 1.83 | 1.91 | 1.040 | same | 8.073% |
| 5I home | img | EA | 0.27 | 0.30 | 1.115 | same | 2.595% |
| 5I home | imglab | EA | 1.63 | 1.75 | 1.069 | same | 1.422% |
| 5I home | imglab50 | EA | 1.61 | 1.64 | 1.017 | same | 2.013% |
| 5I zoom | img | EA | 1.78 | 1.91 | 1.073 | same | 11.367% |
| 5I zoom | imglab | EA | 9.30 | 9.68 | 1.041 | same | 2.021% |
| 5I zoom | imglab50 | EA | 8.73 | 9.00 | 1.031 | same | 6.725% |
| 5I top | img | EA | 0.41 | 0.46 | 1.126 | same | 8.897% |
| 5I top | imglab | EA | 1.72 | 1.66 | 0.964 | same | 4.015% |
| 5I top | imglab50 | EA | 1.27 | 1.31 | 1.034 | slower | 6.581% |
| ftsN_xy1 home | img | EA | 0.27 | 0.30 | 1.095 | same | 1.858% |
| ftsN_xy1 home | imglab | EA | 1.61 | 1.63 | 1.015 | same | 1.298% |
| ftsN_xy1 home | imglab50 | EA | 1.62 | 1.67 | 1.035 | same | 1.548% |
| ftsN_xy1 zoom | img | EA | 1.44 | 1.55 | 1.076 | same | 10.150% |
| ftsN_xy1 zoom | imglab | EA | 5.90 | 6.06 | 1.027 | same | 2.520% |
| ftsN_xy1 zoom | imglab50 | EA | 6.06 | 6.12 | 1.010 | same | 5.683% |
| ftsN_xy1 top | img | EA | 0.38 | 0.41 | 1.088 | same | 1.622% |
| ftsN_xy1 top | imglab | EA | 1.89 | 1.93 | 1.025 | same | 0.857% |
| ftsN_xy1 top | imglab50 | EA | 1.90 | 1.93 | 1.015 | same | 1.092% |
| cells3d home | img | EA | 0.52 | 0.55 | 1.061 | same | 4.197% |
| cells3d home | imglab | EA | 0.92 | 0.97 | 1.056 | same | 3.679% |
| cells3d home | imglab50 | EA | 0.91 | 0.91 | 0.998 | same | 3.855% |
| cells3d zoom | img | EA | 2.81 | 2.97 | 1.057 | same | 5.701% |
| cells3d zoom | imglab | EA | 4.96 | 5.14 | 1.037 | same | 5.051% |
| cells3d zoom | imglab50 | EA | 5.00 | 5.09 | 1.018 | same | 5.192% |
| cells3d top | img | EA | 0.85 | 0.86 | 1.012 | same | 6.316% |
| cells3d top | imglab | EA | 1.32 | 1.36 | 1.031 | same | 4.287% |
| cells3d top | imglab50 | EA | 1.42 | 1.41 | 0.991 | same | 5.179% |

### lab_first

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | imglab | EA | 1.17 | 1.11 | 0.950 | faster | 0.000% |
| dnaA_xy1 home | imglab | MIP | 2.26 | 1.77 | 0.785 | faster | 0.000% |
| dnaA_xy1 home | imglab | mean | 2.25 | 1.77 | 0.787 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | EA | 1.17 | 1.16 | 0.996 | same | 0.000% |
| dnaA_xy1 home | imglab50 | MIP | 2.26 | 2.25 | 0.994 | same | 0.000% |
| dnaA_xy1 home | imglab50 | mean | 2.24 | 2.25 | 1.003 | same | 0.000% |
| dnaA_xy1 zoom | imglab | EA | 5.97 | 5.43 | 0.908 | faster | 0.000% |
| dnaA_xy1 zoom | imglab | MIP | 14.03 | 9.24 | 0.659 | faster | 0.000% |
| dnaA_xy1 zoom | imglab | mean | 14.46 | 9.83 | 0.679 | faster | 0.000% |
| dnaA_xy1 zoom | imglab50 | EA | 7.02 | 7.04 | 1.003 | same | 0.000% |
| dnaA_xy1 zoom | imglab50 | MIP | 16.47 | 16.21 | 0.984 | same | 0.000% |
| dnaA_xy1 zoom | imglab50 | mean | 17.60 | 17.66 | 1.003 | same | 0.000% |
| dnaA_xy1 top | imglab | EA | 1.85 | 1.68 | 0.909 | same | 0.000% |
| dnaA_xy1 top | imglab | MIP | 3.06 | 2.51 | 0.820 | faster | 0.000% |
| dnaA_xy1 top | imglab | mean | 3.71 | 2.66 | 0.718 | same | 0.000% |
| dnaA_xy1 top | imglab50 | EA | 1.83 | 1.84 | 1.004 | same | 0.000% |
| dnaA_xy1 top | imglab50 | MIP | 3.17 | 3.15 | 0.994 | same | 0.000% |
| dnaA_xy1 top | imglab50 | mean | 2.98 | 2.98 | 1.001 | same | 0.000% |
| 5I home | imglab | EA | 1.63 | 1.78 | 1.088 | same | 0.000% |
| 5I home | imglab | MIP | 3.71 | 3.26 | 0.878 | same | 0.000% |
| 5I home | imglab | mean | 3.46 | 2.88 | 0.832 | faster | 0.000% |
| 5I home | imglab50 | EA | 1.61 | 1.71 | 1.062 | same | 0.000% |
| 5I home | imglab50 | MIP | 3.37 | 3.38 | 1.004 | same | 0.000% |
| 5I home | imglab50 | mean | 3.31 | 3.33 | 1.005 | same | 0.000% |
| 5I zoom | imglab | EA | 9.30 | 10.13 | 1.088 | same | 0.000% |
| 5I zoom | imglab | MIP | 21.75 | 17.38 | 0.799 | same | 0.000% |
| 5I zoom | imglab | mean | 19.11 | 14.81 | 0.775 | faster | 0.000% |
| 5I zoom | imglab50 | EA | 8.73 | 8.69 | 0.996 | same | 0.000% |
| 5I zoom | imglab50 | MIP | 18.34 | 19.37 | 1.056 | same | 0.000% |
| 5I zoom | imglab50 | mean | 18.34 | 18.39 | 1.003 | same | 0.000% |
| 5I top | imglab | EA | 1.72 | 1.51 | 0.876 | same | 0.000% |
| 5I top | imglab | MIP | 3.01 | 2.52 | 0.838 | same | 0.000% |
| 5I top | imglab | mean | 3.00 | 2.48 | 0.829 | same | 0.000% |
| 5I top | imglab50 | EA | 1.27 | 1.26 | 0.998 | same | 0.000% |
| 5I top | imglab50 | MIP | 2.61 | 2.72 | 1.045 | same | 0.000% |
| 5I top | imglab50 | mean | 3.23 | 3.22 | 0.997 | same | 0.000% |
| ftsN_xy1 home | imglab | EA | 1.61 | 1.57 | 0.978 | same | 0.000% |
| ftsN_xy1 home | imglab | MIP | 3.66 | 3.00 | 0.820 | faster | 0.000% |
| ftsN_xy1 home | imglab | mean | 3.61 | 2.85 | 0.788 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | EA | 1.62 | 1.64 | 1.018 | same | 0.000% |
| ftsN_xy1 home | imglab50 | MIP | 3.63 | 3.47 | 0.956 | same | 0.000% |
| ftsN_xy1 home | imglab50 | mean | 3.57 | 3.63 | 1.016 | same | 0.000% |
| ftsN_xy1 zoom | imglab | EA | 5.90 | 5.28 | 0.894 | same | 0.000% |
| ftsN_xy1 zoom | imglab | MIP | 17.37 | 8.53 | 0.491 | faster | 0.000% |
| ftsN_xy1 zoom | imglab | mean | 17.74 | 10.10 | 0.569 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | EA | 6.06 | 6.00 | 0.990 | same | 0.000% |
| ftsN_xy1 zoom | imglab50 | MIP | 16.98 | 17.03 | 1.003 | same | 0.000% |
| ftsN_xy1 zoom | imglab50 | mean | 16.91 | 17.09 | 1.011 | same | 0.000% |
| ftsN_xy1 top | imglab | EA | 1.89 | 1.81 | 0.957 | same | 0.000% |
| ftsN_xy1 top | imglab | MIP | 3.45 | 3.23 | 0.936 | same | 0.000% |
| ftsN_xy1 top | imglab | mean | 3.52 | 3.18 | 0.904 | same | 0.000% |
| ftsN_xy1 top | imglab50 | EA | 1.90 | 2.00 | 1.049 | same | 0.000% |
| ftsN_xy1 top | imglab50 | MIP | 3.58 | 3.50 | 0.976 | same | 0.000% |
| ftsN_xy1 top | imglab50 | mean | 3.48 | 3.48 | 0.998 | same | 0.000% |
| cells3d home | imglab | EA | 0.92 | 0.71 | 0.774 | faster | 0.000% |
| cells3d home | imglab | MIP | 1.25 | 0.74 | 0.590 | faster | 0.000% |
| cells3d home | imglab | mean | 1.26 | 0.73 | 0.575 | faster | 0.000% |
| cells3d home | imglab50 | EA | 0.91 | 0.93 | 1.016 | same | 0.000% |
| cells3d home | imglab50 | MIP | 1.25 | 1.30 | 1.038 | same | 0.000% |
| cells3d home | imglab50 | mean | 1.26 | 1.27 | 1.007 | same | 0.000% |
| cells3d zoom | imglab | EA | 4.96 | 3.54 | 0.713 | faster | 0.000% |
| cells3d zoom | imglab | MIP | 6.46 | 3.17 | 0.491 | faster | 0.000% |
| cells3d zoom | imglab | mean | 6.48 | 3.17 | 0.489 | faster | 0.000% |
| cells3d zoom | imglab50 | EA | 5.00 | 4.94 | 0.989 | same | 0.000% |
| cells3d zoom | imglab50 | MIP | 6.57 | 6.58 | 1.002 | same | 0.000% |
| cells3d zoom | imglab50 | mean | 6.48 | 6.60 | 1.017 | same | 0.000% |
| cells3d top | imglab | EA | 1.32 | 1.12 | 0.846 | faster | 0.000% |
| cells3d top | imglab | MIP | 1.27 | 0.91 | 0.719 | faster | 0.000% |
| cells3d top | imglab | mean | 1.38 | 1.04 | 0.751 | faster | 0.000% |
| cells3d top | imglab50 | EA | 1.42 | 1.42 | 0.996 | same | 0.000% |
| cells3d top | imglab50 | MIP | 1.32 | 1.27 | 0.966 | same | 0.000% |
| cells3d top | imglab50 | mean | 1.30 | 1.28 | 0.983 | same | 0.000% |

### clip

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | imglab | EA | 1.17 | 1.22 | 1.044 | same | 3.439% |
| dnaA_xy1 home | imglab | MIP | 2.26 | 1.96 | 0.869 | faster | 3.442% |
| dnaA_xy1 home | imglab | mean | 2.25 | 1.96 | 0.870 | faster | 3.442% |
| dnaA_xy1 home | imglab50 | EA | 1.17 | 1.22 | 1.043 | slower | 5.158% |
| dnaA_xy1 home | imglab50 | MIP | 2.26 | 1.95 | 0.862 | faster | 5.158% |
| dnaA_xy1 home | imglab50 | mean | 2.24 | 1.96 | 0.875 | faster | 5.158% |
| dnaA_xy1 zoom | imglab | EA | 5.97 | 6.27 | 1.050 | slower | 34.047% |
| dnaA_xy1 zoom | imglab | MIP | 14.03 | 10.77 | 0.768 | faster | 34.064% |
| dnaA_xy1 zoom | imglab | mean | 14.46 | 11.16 | 0.771 | faster | 34.064% |
| dnaA_xy1 zoom | imglab50 | EA | 7.02 | 7.37 | 1.050 | same | 42.960% |
| dnaA_xy1 zoom | imglab50 | MIP | 16.47 | 12.52 | 0.761 | same | 42.960% |
| dnaA_xy1 zoom | imglab50 | mean | 17.60 | 13.62 | 0.774 | faster | 42.960% |
| dnaA_xy1 top | imglab | EA | 1.85 | 1.80 | 0.973 | same | 3.510% |
| dnaA_xy1 top | imglab | MIP | 3.06 | 2.60 | 0.848 | same | 3.511% |
| dnaA_xy1 top | imglab | mean | 3.71 | 3.00 | 0.807 | same | 3.511% |
| dnaA_xy1 top | imglab50 | EA | 1.83 | 1.79 | 0.975 | same | 12.076% |
| dnaA_xy1 top | imglab50 | MIP | 3.17 | 2.67 | 0.842 | same | 12.076% |
| dnaA_xy1 top | imglab50 | mean | 2.98 | 2.54 | 0.853 | same | 12.076% |
| 5I home | imglab | EA | 1.63 | 1.72 | 1.053 | same | 1.216% |
| 5I home | imglab | MIP | 3.71 | 3.66 | 0.985 | same | 1.217% |
| 5I home | imglab | mean | 3.46 | 3.09 | 0.893 | faster | 1.217% |
| 5I home | imglab50 | EA | 1.61 | 1.77 | 1.100 | same | 3.770% |
| 5I home | imglab50 | MIP | 3.37 | 3.06 | 0.908 | same | 3.770% |
| 5I home | imglab50 | mean | 3.31 | 3.04 | 0.918 | same | 3.770% |
| 5I zoom | imglab | EA | 9.30 | 9.29 | 0.999 | same | 10.555% |
| 5I zoom | imglab | MIP | 21.75 | 17.49 | 0.804 | same | 10.570% |
| 5I zoom | imglab | mean | 19.11 | 16.41 | 0.859 | same | 10.570% |
| 5I zoom | imglab50 | EA | 8.73 | 8.96 | 1.026 | same | 29.904% |
| 5I zoom | imglab50 | MIP | 18.34 | 15.89 | 0.866 | same | 29.904% |
| 5I zoom | imglab50 | mean | 18.34 | 16.14 | 0.880 | same | 29.904% |
| 5I top | imglab | EA | 1.72 | 1.64 | 0.955 | same | 0.478% |
| 5I top | imglab | MIP | 3.01 | 2.58 | 0.858 | same | 0.479% |
| 5I top | imglab | mean | 3.00 | 2.68 | 0.893 | same | 0.479% |
| 5I top | imglab50 | EA | 1.27 | 1.26 | 0.992 | same | 11.312% |
| 5I top | imglab50 | MIP | 2.61 | 2.39 | 0.917 | same | 11.312% |
| 5I top | imglab50 | mean | 3.23 | 2.76 | 0.852 | faster | 11.312% |
| ftsN_xy1 home | imglab | EA | 1.61 | 1.76 | 1.095 | same | 2.900% |
| ftsN_xy1 home | imglab | MIP | 3.66 | 3.22 | 0.880 | same | 2.900% |
| ftsN_xy1 home | imglab | mean | 3.61 | 3.08 | 0.853 | faster | 2.900% |
| ftsN_xy1 home | imglab50 | EA | 1.62 | 1.75 | 1.086 | same | 3.526% |
| ftsN_xy1 home | imglab50 | MIP | 3.63 | 3.02 | 0.832 | same | 3.526% |
| ftsN_xy1 home | imglab50 | mean | 3.57 | 3.13 | 0.878 | same | 3.526% |
| ftsN_xy1 zoom | imglab | EA | 5.90 | 6.14 | 1.040 | same | 34.959% |
| ftsN_xy1 zoom | imglab | MIP | 17.37 | 10.05 | 0.579 | faster | 34.969% |
| ftsN_xy1 zoom | imglab | mean | 17.74 | 9.78 | 0.551 | faster | 34.969% |
| ftsN_xy1 zoom | imglab50 | EA | 6.06 | 6.26 | 1.032 | same | 41.482% |
| ftsN_xy1 zoom | imglab50 | MIP | 16.98 | 10.08 | 0.594 | faster | 41.482% |
| ftsN_xy1 zoom | imglab50 | mean | 16.91 | 10.06 | 0.595 | faster | 41.482% |
| ftsN_xy1 top | imglab | EA | 1.89 | 1.94 | 1.028 | same | 1.310% |
| ftsN_xy1 top | imglab | MIP | 3.45 | 3.44 | 0.998 | same | 1.311% |
| ftsN_xy1 top | imglab | mean | 3.52 | 3.39 | 0.965 | same | 1.311% |
| ftsN_xy1 top | imglab50 | EA | 1.90 | 1.94 | 1.020 | same | 3.891% |
| ftsN_xy1 top | imglab50 | MIP | 3.58 | 3.61 | 1.008 | same | 3.891% |
| ftsN_xy1 top | imglab50 | mean | 3.48 | 3.45 | 0.989 | same | 3.891% |
| cells3d home | imglab | EA | 0.92 | 0.93 | 1.010 | same | 8.144% |
| cells3d home | imglab | MIP | 1.25 | 0.93 | 0.744 | faster | 8.150% |
| cells3d home | imglab | mean | 1.26 | 0.95 | 0.751 | faster | 8.150% |
| cells3d home | imglab50 | EA | 0.91 | 0.89 | 0.976 | same | 9.333% |
| cells3d home | imglab50 | MIP | 1.25 | 0.96 | 0.769 | faster | 9.333% |
| cells3d home | imglab50 | mean | 1.26 | 0.93 | 0.737 | faster | 9.333% |
| cells3d zoom | imglab | EA | 4.96 | 4.58 | 0.922 | faster | 57.421% |
| cells3d zoom | imglab | MIP | 6.46 | 4.44 | 0.687 | faster | 57.423% |
| cells3d zoom | imglab | mean | 6.48 | 4.56 | 0.703 | faster | 57.423% |
| cells3d zoom | imglab50 | EA | 5.00 | 4.60 | 0.921 | same | 57.871% |
| cells3d zoom | imglab50 | MIP | 6.57 | 4.40 | 0.670 | faster | 57.871% |
| cells3d zoom | imglab50 | mean | 6.48 | 4.41 | 0.680 | faster | 57.871% |
| cells3d top | imglab | EA | 1.32 | 1.23 | 0.929 | same | 17.645% |
| cells3d top | imglab | MIP | 1.27 | 1.07 | 0.840 | faster | 17.645% |
| cells3d top | imglab | mean | 1.38 | 1.14 | 0.822 | same | 17.645% |
| cells3d top | imglab50 | EA | 1.42 | 1.24 | 0.867 | same | 17.934% |
| cells3d top | imglab50 | MIP | 1.32 | 1.08 | 0.815 | faster | 17.934% |
| cells3d top | imglab50 | mean | 1.30 | 1.11 | 0.851 | faster | 17.934% |

### skip4

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.50 | 1.645 | slower | 0.000% |
| dnaA_xy1 home | img | MIP | 1.35 | 1.30 | 0.962 | faster | 0.000% |
| dnaA_xy1 home | img | mean | 1.37 | 1.79 | 1.314 | slower | 0.000% |
| dnaA_xy1 home | imglab | EA | 1.17 | 0.76 | 0.650 | faster | 0.000% |
| dnaA_xy1 home | imglab | MIP | 2.26 | 1.58 | 0.698 | faster | 0.000% |
| dnaA_xy1 home | imglab | mean | 2.25 | 2.05 | 0.912 | faster | 0.000% |
| dnaA_xy1 home | lab | MIP | 0.91 | 0.28 | 0.307 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | EA | 1.17 | 0.76 | 0.650 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | MIP | 2.26 | 1.58 | 0.697 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | mean | 2.24 | 2.05 | 0.913 | faster | 0.000% |
| dnaA_xy1 zoom | img | EA | 1.19 | 1.91 | 1.606 | slower | 0.000% |
| dnaA_xy1 zoom | img | MIP | 7.95 | 5.62 | 0.707 | faster | 0.002% |
| dnaA_xy1 zoom | img | mean | 8.34 | 10.87 | 1.304 | slower | 0.000% |
| dnaA_xy1 zoom | imglab | EA | 5.97 | 3.57 | 0.598 | faster | 0.001% |
| dnaA_xy1 zoom | imglab | MIP | 14.03 | 7.87 | 0.561 | faster | 0.001% |
| dnaA_xy1 zoom | imglab | mean | 14.46 | 13.74 | 0.950 | same | 0.001% |
| dnaA_xy1 zoom | lab | MIP | 5.26 | 1.69 | 0.320 | faster | 0.001% |
| dnaA_xy1 zoom | imglab50 | EA | 7.02 | 4.15 | 0.591 | faster | 0.001% |
| dnaA_xy1 zoom | imglab50 | MIP | 16.47 | 9.62 | 0.584 | faster | 0.002% |
| dnaA_xy1 zoom | imglab50 | mean | 17.60 | 16.68 | 0.948 | same | 0.001% |
| dnaA_xy1 top | img | EA | 0.55 | 0.91 | 1.643 | slower | 0.000% |
| dnaA_xy1 top | img | MIP | 1.92 | 1.85 | 0.965 | same | 0.000% |
| dnaA_xy1 top | img | mean | 1.97 | 2.64 | 1.339 | same | 0.000% |
| dnaA_xy1 top | imglab | EA | 1.85 | 1.40 | 0.759 | same | 0.000% |
| dnaA_xy1 top | imglab | MIP | 3.06 | 2.25 | 0.734 | faster | 0.000% |
| dnaA_xy1 top | imglab | mean | 3.71 | 3.17 | 0.853 | same | 0.000% |
| dnaA_xy1 top | lab | MIP | 1.09 | 0.36 | 0.325 | faster | 0.000% |
| dnaA_xy1 top | imglab50 | EA | 1.83 | 1.40 | 0.762 | same | 0.000% |
| dnaA_xy1 top | imglab50 | MIP | 3.17 | 2.27 | 0.716 | faster | 0.000% |
| dnaA_xy1 top | imglab50 | mean | 2.98 | 2.73 | 0.917 | same | 0.000% |
| 5I home | img | EA | 0.27 | 0.43 | 1.594 | slower | 0.000% |
| 5I home | img | MIP | 2.02 | 2.49 | 1.233 | same | 0.001% |
| 5I home | img | mean | 1.97 | 2.61 | 1.323 | same | 0.000% |
| 5I home | imglab | EA | 1.63 | 0.93 | 0.570 | faster | 0.000% |
| 5I home | imglab | MIP | 3.71 | 3.24 | 0.874 | same | 0.001% |
| 5I home | imglab | mean | 3.46 | 3.01 | 0.870 | faster | 0.000% |
| 5I home | lab | MIP | 1.26 | 0.37 | 0.291 | faster | 0.000% |
| 5I home | imglab50 | EA | 1.61 | 0.91 | 0.566 | faster | 0.000% |
| 5I home | imglab50 | MIP | 3.37 | 2.89 | 0.859 | same | 0.001% |
| 5I home | imglab50 | mean | 3.31 | 2.90 | 0.874 | faster | 0.000% |
| 5I zoom | img | EA | 1.78 | 3.10 | 1.738 | same | 0.000% |
| 5I zoom | img | MIP | 11.51 | 12.17 | 1.057 | same | 0.005% |
| 5I zoom | img | mean | 12.49 | 16.98 | 1.359 | slower | 0.000% |
| 5I zoom | imglab | EA | 9.30 | 4.49 | 0.483 | faster | 0.000% |
| 5I zoom | imglab | MIP | 21.75 | 15.17 | 0.698 | faster | 0.004% |
| 5I zoom | imglab | mean | 19.11 | 17.55 | 0.918 | same | 0.000% |
| 5I zoom | lab | MIP | 7.24 | 1.97 | 0.272 | faster | 0.000% |
| 5I zoom | imglab50 | EA | 8.73 | 4.37 | 0.500 | faster | 0.000% |
| 5I zoom | imglab50 | MIP | 18.34 | 14.31 | 0.780 | faster | 0.005% |
| 5I zoom | imglab50 | mean | 18.34 | 16.68 | 0.909 | same | 0.000% |
| 5I top | img | EA | 0.41 | 0.67 | 1.630 | slower | 0.000% |
| 5I top | img | MIP | 1.84 | 2.28 | 1.242 | same | 0.000% |
| 5I top | img | mean | 1.82 | 2.73 | 1.505 | slower | 0.000% |
| 5I top | imglab | EA | 1.72 | 1.11 | 0.643 | faster | 0.000% |
| 5I top | imglab | MIP | 3.01 | 2.67 | 0.885 | same | 0.000% |
| 5I top | imglab | mean | 3.00 | 2.73 | 0.912 | same | 0.000% |
| 5I top | lab | MIP | 1.10 | 0.33 | 0.299 | faster | 0.000% |
| 5I top | imglab50 | EA | 1.27 | 0.83 | 0.659 | faster | 0.000% |
| 5I top | imglab50 | MIP | 2.61 | 2.42 | 0.928 | same | 0.000% |
| 5I top | imglab50 | mean | 3.23 | 2.93 | 0.906 | same | 0.000% |
| ftsN_xy1 home | img | EA | 0.27 | 0.46 | 1.686 | slower | 0.000% |
| ftsN_xy1 home | img | MIP | 1.67 | 1.38 | 0.823 | faster | 0.000% |
| ftsN_xy1 home | img | mean | 1.70 | 2.23 | 1.313 | slower | 0.000% |
| ftsN_xy1 home | imglab | EA | 1.61 | 0.95 | 0.590 | faster | 0.000% |
| ftsN_xy1 home | imglab | MIP | 3.66 | 2.34 | 0.640 | faster | 0.000% |
| ftsN_xy1 home | imglab | mean | 3.61 | 3.22 | 0.890 | same | 0.000% |
| ftsN_xy1 home | lab | MIP | 1.28 | 0.42 | 0.324 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | EA | 1.62 | 0.96 | 0.594 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | MIP | 3.63 | 2.21 | 0.608 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | mean | 3.57 | 3.29 | 0.923 | same | 0.000% |
| ftsN_xy1 zoom | img | EA | 1.44 | 2.76 | 1.918 | slower | 0.000% |
| ftsN_xy1 zoom | img | MIP | 13.89 | 8.75 | 0.630 | faster | 0.001% |
| ftsN_xy1 zoom | img | mean | 12.42 | 16.29 | 1.311 | slower | 0.000% |
| ftsN_xy1 zoom | imglab | EA | 5.90 | 3.71 | 0.628 | faster | 0.000% |
| ftsN_xy1 zoom | imglab | MIP | 17.37 | 9.05 | 0.521 | faster | 0.001% |
| ftsN_xy1 zoom | imglab | mean | 17.74 | 18.00 | 1.014 | same | 0.000% |
| ftsN_xy1 zoom | lab | MIP | 4.98 | 1.58 | 0.317 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | EA | 6.06 | 3.77 | 0.623 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | MIP | 16.98 | 8.98 | 0.529 | faster | 0.001% |
| ftsN_xy1 zoom | imglab50 | mean | 16.91 | 17.68 | 1.046 | same | 0.000% |
| ftsN_xy1 top | img | EA | 0.38 | 0.62 | 1.619 | slower | 0.000% |
| ftsN_xy1 top | img | MIP | 1.88 | 2.64 | 1.407 | slower | 0.000% |
| ftsN_xy1 top | img | mean | 1.95 | 2.59 | 1.333 | same | 0.000% |
| ftsN_xy1 top | imglab | EA | 1.89 | 1.18 | 0.625 | faster | 0.000% |
| ftsN_xy1 top | imglab | MIP | 3.45 | 3.24 | 0.939 | same | 0.000% |
| ftsN_xy1 top | imglab | mean | 3.52 | 3.16 | 0.898 | same | 0.000% |
| ftsN_xy1 top | lab | MIP | 1.46 | 0.53 | 0.367 | faster | 0.000% |
| ftsN_xy1 top | imglab50 | EA | 1.90 | 1.18 | 0.620 | faster | 0.000% |
| ftsN_xy1 top | imglab50 | MIP | 3.58 | 3.38 | 0.943 | same | 0.000% |
| ftsN_xy1 top | imglab50 | mean | 3.48 | 3.22 | 0.925 | same | 0.000% |
| cells3d home | img | EA | 0.52 | 0.87 | 1.669 | slower | 0.000% |
| cells3d home | img | MIP | 0.84 | 0.90 | 1.060 | same | 0.000% |
| cells3d home | img | mean | 0.83 | 1.03 | 1.243 | slower | 0.000% |
| cells3d home | imglab | EA | 0.92 | 1.10 | 1.201 | same | 0.000% |
| cells3d home | imglab | MIP | 1.25 | 1.10 | 0.880 | faster | 0.000% |
| cells3d home | imglab | mean | 1.26 | 1.23 | 0.977 | same | 0.000% |
| cells3d home | lab | MIP | 0.41 | 0.23 | 0.554 | faster | 0.000% |
| cells3d home | imglab50 | EA | 0.91 | 1.10 | 1.199 | same | 0.000% |
| cells3d home | imglab50 | MIP | 1.25 | 1.17 | 0.937 | same | 0.000% |
| cells3d home | imglab50 | mean | 1.26 | 1.26 | 0.998 | same | 0.000% |
| cells3d zoom | img | EA | 2.81 | 4.60 | 1.637 | slower | 0.000% |
| cells3d zoom | img | MIP | 4.46 | 4.36 | 0.978 | same | 0.000% |
| cells3d zoom | img | mean | 4.75 | 6.02 | 1.267 | slower | 0.000% |
| cells3d zoom | imglab | EA | 4.96 | 5.49 | 1.107 | same | 0.000% |
| cells3d zoom | imglab | MIP | 6.46 | 5.37 | 0.832 | faster | 0.000% |
| cells3d zoom | imglab | mean | 6.48 | 6.78 | 1.046 | same | 0.000% |
| cells3d zoom | lab | MIP | 2.03 | 0.94 | 0.460 | faster | 0.000% |
| cells3d zoom | imglab50 | EA | 5.00 | 5.50 | 1.100 | same | 0.000% |
| cells3d zoom | imglab50 | MIP | 6.57 | 5.39 | 0.822 | faster | 0.001% |
| cells3d zoom | imglab50 | mean | 6.48 | 6.80 | 1.049 | slower | 0.000% |
| cells3d top | img | EA | 0.85 | 1.36 | 1.603 | slower | 0.000% |
| cells3d top | img | MIP | 0.82 | 1.14 | 1.396 | slower | 0.000% |
| cells3d top | img | mean | 0.79 | 1.03 | 1.304 | slower | 0.000% |
| cells3d top | imglab | EA | 1.32 | 1.65 | 1.244 | same | 0.000% |
| cells3d top | imglab | MIP | 1.27 | 1.39 | 1.098 | same | 0.000% |
| cells3d top | imglab | mean | 1.38 | 1.35 | 0.973 | same | 0.000% |
| cells3d top | lab | MIP | 0.47 | 0.24 | 0.521 | faster | 0.000% |
| cells3d top | imglab50 | EA | 1.42 | 1.75 | 1.229 | same | 0.000% |
| cells3d top | imglab50 | MIP | 1.32 | 1.41 | 1.070 | same | 0.000% |
| cells3d top | imglab50 | mean | 1.30 | 1.34 | 1.028 | same | 0.000% |

### skip8

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.48 | 1.587 | slower | 0.000% |
| dnaA_xy1 home | img | MIP | 1.35 | 1.57 | 1.165 | slower | 0.000% |
| dnaA_xy1 home | img | mean | 1.37 | 1.80 | 1.316 | slower | 0.000% |
| dnaA_xy1 home | imglab | EA | 1.17 | 0.65 | 0.558 | faster | 0.000% |
| dnaA_xy1 home | imglab | MIP | 2.26 | 1.77 | 0.783 | faster | 0.000% |
| dnaA_xy1 home | imglab | mean | 2.25 | 1.95 | 0.866 | faster | 0.000% |
| dnaA_xy1 home | lab | MIP | 0.91 | 0.23 | 0.257 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | EA | 1.17 | 0.65 | 0.559 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | MIP | 2.26 | 1.76 | 0.778 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | mean | 2.24 | 1.96 | 0.873 | faster | 0.000% |
| dnaA_xy1 zoom | img | EA | 1.19 | 1.82 | 1.529 | slower | 0.000% |
| dnaA_xy1 zoom | img | MIP | 7.95 | 6.52 | 0.820 | faster | 0.001% |
| dnaA_xy1 zoom | img | mean | 8.34 | 10.88 | 1.305 | slower | 0.000% |
| dnaA_xy1 zoom | imglab | EA | 5.97 | 3.00 | 0.502 | faster | 0.000% |
| dnaA_xy1 zoom | imglab | MIP | 14.03 | 8.21 | 0.586 | faster | 0.001% |
| dnaA_xy1 zoom | imglab | mean | 14.46 | 13.20 | 0.913 | same | 0.000% |
| dnaA_xy1 zoom | lab | MIP | 5.26 | 1.06 | 0.202 | faster | 0.000% |
| dnaA_xy1 zoom | imglab50 | EA | 7.02 | 4.03 | 0.574 | faster | 0.000% |
| dnaA_xy1 zoom | imglab50 | MIP | 16.47 | 9.57 | 0.581 | faster | 0.001% |
| dnaA_xy1 zoom | imglab50 | mean | 17.60 | 16.00 | 0.909 | same | 0.000% |
| dnaA_xy1 top | img | EA | 0.55 | 0.80 | 1.445 | slower | 0.000% |
| dnaA_xy1 top | img | MIP | 1.92 | 2.03 | 1.058 | same | 0.000% |
| dnaA_xy1 top | img | mean | 1.97 | 2.63 | 1.332 | same | 0.000% |
| dnaA_xy1 top | imglab | EA | 1.85 | 1.19 | 0.645 | faster | 0.000% |
| dnaA_xy1 top | imglab | MIP | 3.06 | 2.14 | 0.697 | faster | 0.000% |
| dnaA_xy1 top | imglab | mean | 3.71 | 3.09 | 0.832 | same | 0.000% |
| dnaA_xy1 top | lab | MIP | 1.09 | 0.23 | 0.211 | faster | 0.000% |
| dnaA_xy1 top | imglab50 | EA | 1.83 | 1.18 | 0.644 | faster | 0.000% |
| dnaA_xy1 top | imglab50 | MIP | 3.17 | 2.31 | 0.728 | faster | 0.000% |
| dnaA_xy1 top | imglab50 | mean | 2.98 | 2.59 | 0.867 | faster | 0.000% |
| 5I home | img | EA | 0.27 | 0.42 | 1.535 | slower | 0.000% |
| 5I home | img | MIP | 2.02 | 2.45 | 1.215 | same | 0.000% |
| 5I home | img | mean | 1.97 | 2.65 | 1.342 | same | 0.000% |
| 5I home | imglab | EA | 1.63 | 0.75 | 0.461 | faster | 0.000% |
| 5I home | imglab | MIP | 3.71 | 2.83 | 0.762 | same | 0.000% |
| 5I home | imglab | mean | 3.46 | 2.84 | 0.820 | faster | 0.000% |
| 5I home | lab | MIP | 1.26 | 0.26 | 0.205 | faster | 0.000% |
| 5I home | imglab50 | EA | 1.61 | 0.71 | 0.438 | faster | 0.000% |
| 5I home | imglab50 | MIP | 3.37 | 2.85 | 0.845 | same | 0.000% |
| 5I home | imglab50 | mean | 3.31 | 2.73 | 0.822 | faster | 0.000% |
| 5I zoom | img | EA | 1.78 | 2.74 | 1.536 | same | 0.000% |
| 5I zoom | img | MIP | 11.51 | 12.05 | 1.046 | same | 0.002% |
| 5I zoom | img | mean | 12.49 | 16.99 | 1.360 | slower | 0.000% |
| 5I zoom | imglab | EA | 9.30 | 3.86 | 0.415 | faster | 0.000% |
| 5I zoom | imglab | MIP | 21.75 | 13.56 | 0.623 | faster | 0.001% |
| 5I zoom | imglab | mean | 19.11 | 16.83 | 0.881 | same | 0.000% |
| 5I zoom | lab | MIP | 7.24 | 1.23 | 0.170 | faster | 0.000% |
| 5I zoom | imglab50 | EA | 8.73 | 3.47 | 0.398 | faster | 0.000% |
| 5I zoom | imglab50 | MIP | 18.34 | 15.19 | 0.828 | faster | 0.002% |
| 5I zoom | imglab50 | mean | 18.34 | 15.82 | 0.862 | faster | 0.000% |
| 5I top | img | EA | 0.41 | 0.63 | 1.535 | slower | 0.000% |
| 5I top | img | MIP | 1.84 | 2.20 | 1.193 | same | 0.000% |
| 5I top | img | mean | 1.82 | 2.37 | 1.303 | slower | 0.000% |
| 5I top | imglab | EA | 1.72 | 0.86 | 0.498 | faster | 0.000% |
| 5I top | imglab | MIP | 3.01 | 2.40 | 0.796 | faster | 0.000% |
| 5I top | imglab | mean | 3.00 | 2.57 | 0.858 | same | 0.000% |
| 5I top | lab | MIP | 1.10 | 0.19 | 0.171 | faster | 0.000% |
| 5I top | imglab50 | EA | 1.27 | 0.68 | 0.536 | faster | 0.000% |
| 5I top | imglab50 | MIP | 2.61 | 2.07 | 0.795 | same | 0.000% |
| 5I top | imglab50 | mean | 3.23 | 2.75 | 0.850 | faster | 0.000% |
| ftsN_xy1 home | img | EA | 0.27 | 0.45 | 1.651 | slower | 0.000% |
| ftsN_xy1 home | img | MIP | 1.67 | 1.72 | 1.025 | same | 0.000% |
| ftsN_xy1 home | img | mean | 1.70 | 2.20 | 1.291 | slower | 0.000% |
| ftsN_xy1 home | imglab | EA | 1.61 | 0.83 | 0.518 | faster | 0.000% |
| ftsN_xy1 home | imglab | MIP | 3.66 | 2.64 | 0.721 | faster | 0.000% |
| ftsN_xy1 home | imglab | mean | 3.61 | 3.18 | 0.881 | faster | 0.000% |
| ftsN_xy1 home | lab | MIP | 1.28 | 0.34 | 0.262 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | EA | 1.62 | 0.84 | 0.517 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | MIP | 3.63 | 2.86 | 0.786 | same | 0.000% |
| ftsN_xy1 home | imglab50 | mean | 3.57 | 3.18 | 0.892 | same | 0.000% |
| ftsN_xy1 zoom | img | EA | 1.44 | 2.29 | 1.589 | slower | 0.000% |
| ftsN_xy1 zoom | img | MIP | 13.89 | 8.22 | 0.592 | faster | 0.001% |
| ftsN_xy1 zoom | img | mean | 12.42 | 16.13 | 1.299 | slower | 0.000% |
| ftsN_xy1 zoom | imglab | EA | 5.90 | 3.16 | 0.536 | faster | 0.000% |
| ftsN_xy1 zoom | imglab | MIP | 17.37 | 8.87 | 0.511 | faster | 0.000% |
| ftsN_xy1 zoom | imglab | mean | 17.74 | 17.65 | 0.995 | same | 0.000% |
| ftsN_xy1 zoom | lab | MIP | 4.98 | 1.06 | 0.212 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | EA | 6.06 | 3.27 | 0.539 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | MIP | 16.98 | 8.84 | 0.520 | faster | 0.001% |
| ftsN_xy1 zoom | imglab50 | mean | 16.91 | 17.43 | 1.031 | same | 0.000% |
| ftsN_xy1 top | img | EA | 0.38 | 0.58 | 1.532 | slower | 0.000% |
| ftsN_xy1 top | img | MIP | 1.88 | 2.93 | 1.558 | slower | 0.000% |
| ftsN_xy1 top | img | mean | 1.95 | 2.67 | 1.371 | same | 0.000% |
| ftsN_xy1 top | imglab | EA | 1.89 | 1.01 | 0.537 | faster | 0.000% |
| ftsN_xy1 top | imglab | MIP | 3.45 | 3.39 | 0.983 | same | 0.000% |
| ftsN_xy1 top | imglab | mean | 3.52 | 2.99 | 0.850 | same | 0.000% |
| ftsN_xy1 top | lab | MIP | 1.46 | 0.41 | 0.284 | faster | 0.000% |
| ftsN_xy1 top | imglab50 | EA | 1.90 | 1.07 | 0.561 | faster | 0.000% |
| ftsN_xy1 top | imglab50 | MIP | 3.58 | 3.35 | 0.935 | same | 0.000% |
| ftsN_xy1 top | imglab50 | mean | 3.48 | 3.05 | 0.875 | same | 0.000% |
| cells3d home | img | EA | 0.52 | 0.84 | 1.610 | slower | 0.000% |
| cells3d home | img | MIP | 0.84 | 0.93 | 1.097 | same | 0.000% |
| cells3d home | img | mean | 0.83 | 1.03 | 1.239 | slower | 0.000% |
| cells3d home | imglab | EA | 0.92 | 1.11 | 1.212 | same | 0.000% |
| cells3d home | imglab | MIP | 1.25 | 1.20 | 0.956 | same | 0.000% |
| cells3d home | imglab | mean | 1.26 | 1.27 | 1.007 | same | 0.000% |
| cells3d home | lab | MIP | 0.41 | 0.28 | 0.689 | faster | 0.000% |
| cells3d home | imglab50 | EA | 0.91 | 1.11 | 1.212 | same | 0.000% |
| cells3d home | imglab50 | MIP | 1.25 | 1.27 | 1.011 | same | 0.000% |
| cells3d home | imglab50 | mean | 1.26 | 1.31 | 1.041 | same | 0.000% |
| cells3d zoom | img | EA | 2.81 | 4.37 | 1.555 | slower | 0.000% |
| cells3d zoom | img | MIP | 4.46 | 4.51 | 1.011 | same | 0.000% |
| cells3d zoom | img | mean | 4.75 | 5.83 | 1.226 | slower | 0.000% |
| cells3d zoom | imglab | EA | 4.96 | 5.51 | 1.111 | same | 0.000% |
| cells3d zoom | imglab | MIP | 6.46 | 5.72 | 0.886 | faster | 0.000% |
| cells3d zoom | imglab | mean | 6.48 | 7.20 | 1.110 | same | 0.000% |
| cells3d zoom | lab | MIP | 2.03 | 1.15 | 0.568 | faster | 0.000% |
| cells3d zoom | imglab50 | EA | 5.00 | 5.92 | 1.185 | same | 0.000% |
| cells3d zoom | imglab50 | MIP | 6.57 | 5.76 | 0.878 | same | 0.000% |
| cells3d zoom | imglab50 | mean | 6.48 | 7.01 | 1.081 | slower | 0.000% |
| cells3d top | img | EA | 0.85 | 1.38 | 1.624 | slower | 0.000% |
| cells3d top | img | MIP | 0.82 | 1.04 | 1.274 | slower | 0.000% |
| cells3d top | img | mean | 0.79 | 1.01 | 1.288 | slower | 0.000% |
| cells3d top | imglab | EA | 1.32 | 1.63 | 1.235 | same | 0.000% |
| cells3d top | imglab | MIP | 1.27 | 1.32 | 1.043 | same | 0.000% |
| cells3d top | imglab | mean | 1.38 | 1.38 | 0.997 | same | 0.000% |
| cells3d top | lab | MIP | 0.47 | 0.28 | 0.601 | faster | 0.000% |
| cells3d top | imglab50 | EA | 1.42 | 1.72 | 1.209 | same | 0.000% |
| cells3d top | imglab50 | MIP | 1.32 | 1.37 | 1.037 | same | 0.000% |
| cells3d top | imglab50 | mean | 1.30 | 1.33 | 1.025 | same | 0.000% |

### skip16

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.46 | 1.511 | slower | 0.000% |
| dnaA_xy1 home | img | MIP | 1.35 | 1.62 | 1.196 | slower | 0.000% |
| dnaA_xy1 home | img | mean | 1.37 | 1.80 | 1.316 | slower | 0.000% |
| dnaA_xy1 home | imglab | EA | 1.17 | 0.63 | 0.537 | faster | 0.000% |
| dnaA_xy1 home | imglab | MIP | 2.26 | 1.80 | 0.798 | faster | 0.000% |
| dnaA_xy1 home | imglab | mean | 2.25 | 1.94 | 0.863 | faster | 0.000% |
| dnaA_xy1 home | lab | MIP | 0.91 | 0.25 | 0.278 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | EA | 1.17 | 0.63 | 0.537 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | MIP | 2.26 | 1.78 | 0.788 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | mean | 2.24 | 1.95 | 0.870 | faster | 0.000% |
| dnaA_xy1 zoom | img | EA | 1.19 | 1.77 | 1.488 | slower | 0.000% |
| dnaA_xy1 zoom | img | MIP | 7.95 | 7.44 | 0.936 | faster | 0.001% |
| dnaA_xy1 zoom | img | mean | 8.34 | 10.84 | 1.299 | slower | 0.000% |
| dnaA_xy1 zoom | imglab | EA | 5.97 | 2.99 | 0.501 | faster | 0.000% |
| dnaA_xy1 zoom | imglab | MIP | 14.03 | 9.45 | 0.673 | faster | 0.001% |
| dnaA_xy1 zoom | imglab | mean | 14.46 | 13.31 | 0.920 | faster | 0.000% |
| dnaA_xy1 zoom | lab | MIP | 5.26 | 1.30 | 0.247 | faster | 0.000% |
| dnaA_xy1 zoom | imglab50 | EA | 7.02 | 3.46 | 0.492 | faster | 0.000% |
| dnaA_xy1 zoom | imglab50 | MIP | 16.47 | 10.67 | 0.648 | faster | 0.001% |
| dnaA_xy1 zoom | imglab50 | mean | 17.60 | 16.08 | 0.913 | faster | 0.000% |
| dnaA_xy1 top | img | EA | 0.55 | 0.83 | 1.491 | slower | 0.000% |
| dnaA_xy1 top | img | MIP | 1.92 | 1.99 | 1.039 | same | 0.000% |
| dnaA_xy1 top | img | mean | 1.97 | 2.67 | 1.354 | same | 0.000% |
| dnaA_xy1 top | imglab | EA | 1.85 | 1.14 | 0.616 | faster | 0.000% |
| dnaA_xy1 top | imglab | MIP | 3.06 | 2.08 | 0.680 | faster | 0.000% |
| dnaA_xy1 top | imglab | mean | 3.71 | 2.99 | 0.806 | same | 0.000% |
| dnaA_xy1 top | lab | MIP | 1.09 | 0.22 | 0.204 | faster | 0.000% |
| dnaA_xy1 top | imglab50 | EA | 1.83 | 1.14 | 0.622 | faster | 0.000% |
| dnaA_xy1 top | imglab50 | MIP | 3.17 | 2.22 | 0.701 | faster | 0.000% |
| dnaA_xy1 top | imglab50 | mean | 2.98 | 2.61 | 0.874 | same | 0.000% |
| 5I home | img | EA | 0.27 | 0.40 | 1.460 | slower | 0.000% |
| 5I home | img | MIP | 2.02 | 2.52 | 1.247 | same | 0.000% |
| 5I home | img | mean | 1.97 | 2.95 | 1.496 | same | 0.000% |
| 5I home | imglab | EA | 1.63 | 0.66 | 0.403 | faster | 0.000% |
| 5I home | imglab | MIP | 3.71 | 3.27 | 0.881 | same | 0.000% |
| 5I home | imglab | mean | 3.46 | 2.88 | 0.830 | same | 0.000% |
| 5I home | lab | MIP | 1.26 | 0.31 | 0.243 | faster | 0.000% |
| 5I home | imglab50 | EA | 1.61 | 0.70 | 0.431 | faster | 0.000% |
| 5I home | imglab50 | MIP | 3.37 | 2.69 | 0.799 | faster | 0.000% |
| 5I home | imglab50 | mean | 3.31 | 2.79 | 0.841 | faster | 0.000% |
| 5I zoom | img | EA | 1.78 | 2.59 | 1.452 | same | 0.000% |
| 5I zoom | img | MIP | 11.51 | 12.82 | 1.113 | same | 0.001% |
| 5I zoom | img | mean | 12.49 | 17.80 | 1.425 | slower | 0.000% |
| 5I zoom | imglab | EA | 9.30 | 3.53 | 0.379 | faster | 0.000% |
| 5I zoom | imglab | MIP | 21.75 | 14.48 | 0.666 | faster | 0.001% |
| 5I zoom | imglab | mean | 19.11 | 16.19 | 0.847 | faster | 0.000% |
| 5I zoom | lab | MIP | 7.24 | 1.19 | 0.164 | faster | 0.000% |
| 5I zoom | imglab50 | EA | 8.73 | 3.38 | 0.387 | faster | 0.000% |
| 5I zoom | imglab50 | MIP | 18.34 | 14.23 | 0.776 | same | 0.001% |
| 5I zoom | imglab50 | mean | 18.34 | 15.59 | 0.850 | faster | 0.000% |
| 5I top | img | EA | 0.41 | 0.61 | 1.481 | slower | 0.000% |
| 5I top | img | MIP | 1.84 | 2.25 | 1.221 | same | 0.000% |
| 5I top | img | mean | 1.82 | 2.38 | 1.307 | slower | 0.000% |
| 5I top | imglab | EA | 1.72 | 0.77 | 0.447 | faster | 0.000% |
| 5I top | imglab | MIP | 3.01 | 2.35 | 0.782 | faster | 0.000% |
| 5I top | imglab | mean | 3.00 | 2.93 | 0.976 | same | 0.000% |
| 5I top | lab | MIP | 1.10 | 0.14 | 0.125 | faster | 0.000% |
| 5I top | imglab50 | EA | 1.27 | 0.61 | 0.482 | faster | 0.000% |
| 5I top | imglab50 | MIP | 2.61 | 2.07 | 0.792 | faster | 0.000% |
| 5I top | imglab50 | mean | 3.23 | 2.67 | 0.826 | faster | 0.000% |
| ftsN_xy1 home | img | EA | 0.27 | 0.42 | 1.538 | slower | 0.000% |
| ftsN_xy1 home | img | MIP | 1.67 | 1.88 | 1.121 | slower | 0.000% |
| ftsN_xy1 home | img | mean | 1.70 | 2.22 | 1.308 | slower | 0.000% |
| ftsN_xy1 home | imglab | EA | 1.61 | 0.80 | 0.497 | faster | 0.000% |
| ftsN_xy1 home | imglab | MIP | 3.66 | 2.84 | 0.776 | faster | 0.000% |
| ftsN_xy1 home | imglab | mean | 3.61 | 3.17 | 0.878 | same | 0.000% |
| ftsN_xy1 home | lab | MIP | 1.28 | 0.38 | 0.296 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | EA | 1.62 | 0.83 | 0.515 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | MIP | 3.63 | 2.69 | 0.741 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | mean | 3.57 | 3.15 | 0.883 | same | 0.000% |
| ftsN_xy1 zoom | img | EA | 1.44 | 2.28 | 1.583 | slower | 0.000% |
| ftsN_xy1 zoom | img | MIP | 13.89 | 8.50 | 0.612 | faster | 0.000% |
| ftsN_xy1 zoom | img | mean | 12.42 | 16.12 | 1.298 | slower | 0.000% |
| ftsN_xy1 zoom | imglab | EA | 5.90 | 3.10 | 0.525 | faster | 0.000% |
| ftsN_xy1 zoom | imglab | MIP | 17.37 | 9.88 | 0.569 | faster | 0.000% |
| ftsN_xy1 zoom | imglab | mean | 17.74 | 17.41 | 0.981 | same | 0.000% |
| ftsN_xy1 zoom | lab | MIP | 4.98 | 1.22 | 0.245 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | EA | 6.06 | 3.24 | 0.534 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | MIP | 16.98 | 9.48 | 0.558 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | mean | 16.91 | 18.23 | 1.078 | same | 0.000% |
| ftsN_xy1 top | img | EA | 0.38 | 0.56 | 1.472 | slower | 0.000% |
| ftsN_xy1 top | img | MIP | 1.88 | 2.61 | 1.390 | slower | 0.000% |
| ftsN_xy1 top | img | mean | 1.95 | 2.61 | 1.340 | same | 0.000% |
| ftsN_xy1 top | imglab | EA | 1.89 | 0.97 | 0.512 | faster | 0.000% |
| ftsN_xy1 top | imglab | MIP | 3.45 | 3.08 | 0.893 | same | 0.000% |
| ftsN_xy1 top | imglab | mean | 3.52 | 3.02 | 0.859 | same | 0.000% |
| ftsN_xy1 top | lab | MIP | 1.46 | 0.41 | 0.280 | faster | 0.000% |
| ftsN_xy1 top | imglab50 | EA | 1.90 | 1.13 | 0.594 | faster | 0.000% |
| ftsN_xy1 top | imglab50 | MIP | 3.58 | 2.99 | 0.835 | faster | 0.000% |
| ftsN_xy1 top | imglab50 | mean | 3.48 | 2.96 | 0.850 | same | 0.000% |
| cells3d home | img | EA | 0.52 | 0.80 | 1.537 | slower | 0.000% |
| cells3d home | img | MIP | 0.84 | 0.96 | 1.138 | same | 0.000% |
| cells3d home | img | mean | 0.83 | 1.03 | 1.238 | slower | 0.000% |
| cells3d home | imglab | EA | 0.92 | 1.24 | 1.345 | same | 0.000% |
| cells3d home | imglab | MIP | 1.25 | 1.37 | 1.092 | same | 0.000% |
| cells3d home | imglab | mean | 1.26 | 1.40 | 1.105 | same | 0.000% |
| cells3d home | lab | MIP | 0.41 | 0.37 | 0.890 | faster | 0.000% |
| cells3d home | imglab50 | EA | 0.91 | 1.15 | 1.258 | slower | 0.000% |
| cells3d home | imglab50 | MIP | 1.25 | 1.38 | 1.104 | slower | 0.000% |
| cells3d home | imglab50 | mean | 1.26 | 1.40 | 1.113 | same | 0.000% |
| cells3d zoom | img | EA | 2.81 | 4.42 | 1.574 | slower | 0.000% |
| cells3d zoom | img | MIP | 4.46 | 4.97 | 1.116 | slower | 0.000% |
| cells3d zoom | img | mean | 4.75 | 6.66 | 1.401 | slower | 0.000% |
| cells3d zoom | imglab | EA | 4.96 | 6.03 | 1.214 | slower | 0.000% |
| cells3d zoom | imglab | MIP | 6.46 | 6.90 | 1.069 | same | 0.000% |
| cells3d zoom | imglab | mean | 6.48 | 7.66 | 1.181 | slower | 0.000% |
| cells3d zoom | lab | MIP | 2.03 | 1.76 | 0.867 | same | 0.000% |
| cells3d zoom | imglab50 | EA | 5.00 | 6.08 | 1.216 | slower | 0.000% |
| cells3d zoom | imglab50 | MIP | 6.57 | 7.78 | 1.185 | same | 0.000% |
| cells3d zoom | imglab50 | mean | 6.48 | 7.80 | 1.203 | slower | 0.000% |
| cells3d top | img | EA | 0.85 | 1.24 | 1.460 | slower | 0.000% |
| cells3d top | img | MIP | 0.82 | 1.05 | 1.285 | slower | 0.000% |
| cells3d top | img | mean | 0.79 | 1.01 | 1.279 | slower | 0.000% |
| cells3d top | imglab | EA | 1.32 | 1.75 | 1.321 | same | 0.000% |
| cells3d top | imglab | MIP | 1.27 | 1.41 | 1.114 | slower | 0.000% |
| cells3d top | imglab | mean | 1.38 | 1.57 | 1.136 | same | 0.000% |
| cells3d top | lab | MIP | 0.47 | 0.40 | 0.859 | faster | 0.000% |
| cells3d top | imglab50 | EA | 1.42 | 1.75 | 1.227 | same | 0.000% |
| cells3d top | imglab50 | MIP | 1.32 | 1.50 | 1.135 | slower | 0.000% |
| cells3d top | imglab50 | mean | 1.30 | 1.43 | 1.096 | same | 0.000% |

### lut

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | imglab | EA | 1.17 | 1.18 | 1.014 | same | 0.000% |
| dnaA_xy1 home | imglab | MIP | 2.26 | 2.29 | 1.016 | same | 0.000% |
| dnaA_xy1 home | imglab | mean | 2.25 | 2.28 | 1.013 | same | 0.000% |
| dnaA_xy1 home | lab | MIP | 0.91 | 0.91 | 1.001 | same | 0.000% |
| dnaA_xy1 home | imglab50 | EA | 1.17 | 1.18 | 1.012 | same | 0.000% |
| dnaA_xy1 home | imglab50 | MIP | 2.26 | 2.29 | 1.014 | same | 0.000% |
| dnaA_xy1 home | imglab50 | mean | 2.24 | 2.28 | 1.018 | same | 0.000% |
| dnaA_xy1 zoom | imglab | EA | 5.97 | 6.08 | 1.019 | same | 0.000% |
| dnaA_xy1 zoom | imglab | MIP | 14.03 | 13.92 | 0.992 | same | 0.000% |
| dnaA_xy1 zoom | imglab | mean | 14.46 | 14.69 | 1.016 | same | 0.000% |
| dnaA_xy1 zoom | lab | MIP | 5.26 | 5.86 | 1.114 | same | 0.000% |
| dnaA_xy1 zoom | imglab50 | EA | 7.02 | 7.10 | 1.012 | same | 0.000% |
| dnaA_xy1 zoom | imglab50 | MIP | 16.47 | 16.47 | 1.000 | same | 0.000% |
| dnaA_xy1 zoom | imglab50 | mean | 17.60 | 17.62 | 1.001 | same | 0.000% |
| dnaA_xy1 top | imglab | EA | 1.85 | 1.88 | 1.017 | same | 0.000% |
| dnaA_xy1 top | imglab | MIP | 3.06 | 3.14 | 1.026 | same | 0.000% |
| dnaA_xy1 top | imglab | mean | 3.71 | 3.44 | 0.926 | same | 0.000% |
| dnaA_xy1 top | lab | MIP | 1.09 | 1.10 | 1.011 | same | 0.000% |
| dnaA_xy1 top | imglab50 | EA | 1.83 | 1.88 | 1.025 | same | 0.000% |
| dnaA_xy1 top | imglab50 | MIP | 3.17 | 3.13 | 0.986 | same | 0.000% |
| dnaA_xy1 top | imglab50 | mean | 2.98 | 3.02 | 1.014 | same | 0.000% |
| 5I home | imglab | EA | 1.63 | 1.65 | 1.012 | same | 0.000% |
| 5I home | imglab | MIP | 3.71 | 3.52 | 0.949 | same | 0.000% |
| 5I home | imglab | mean | 3.46 | 3.44 | 0.992 | same | 0.000% |
| 5I home | lab | MIP | 1.26 | 1.30 | 1.032 | same | 0.000% |
| 5I home | imglab50 | EA | 1.61 | 1.67 | 1.033 | same | 0.000% |
| 5I home | imglab50 | MIP | 3.37 | 3.44 | 1.022 | same | 0.000% |
| 5I home | imglab50 | mean | 3.31 | 3.46 | 1.044 | same | 0.000% |
| 5I zoom | imglab | EA | 9.30 | 9.72 | 1.045 | same | 0.000% |
| 5I zoom | imglab | MIP | 21.75 | 19.69 | 0.905 | same | 0.000% |
| 5I zoom | imglab | mean | 19.11 | 19.26 | 1.008 | same | 0.000% |
| 5I zoom | lab | MIP | 7.24 | 7.17 | 0.991 | same | 0.000% |
| 5I zoom | imglab50 | EA | 8.73 | 8.68 | 0.995 | same | 0.000% |
| 5I zoom | imglab50 | MIP | 18.34 | 19.04 | 1.038 | same | 0.000% |
| 5I zoom | imglab50 | mean | 18.34 | 18.82 | 1.026 | same | 0.000% |
| 5I top | imglab | EA | 1.72 | 1.63 | 0.948 | same | 0.000% |
| 5I top | imglab | MIP | 3.01 | 3.12 | 1.036 | same | 0.000% |
| 5I top | imglab | mean | 3.00 | 3.10 | 1.035 | same | 0.000% |
| 5I top | lab | MIP | 1.10 | 1.12 | 1.014 | same | 0.000% |
| 5I top | imglab50 | EA | 1.27 | 1.29 | 1.020 | slower | 0.000% |
| 5I top | imglab50 | MIP | 2.61 | 2.68 | 1.026 | same | 0.000% |
| 5I top | imglab50 | mean | 3.23 | 3.58 | 1.106 | same | 0.000% |
| ftsN_xy1 home | imglab | EA | 1.61 | 1.66 | 1.035 | same | 0.000% |
| ftsN_xy1 home | imglab | MIP | 3.66 | 3.74 | 1.022 | same | 0.000% |
| ftsN_xy1 home | imglab | mean | 3.61 | 3.59 | 0.993 | same | 0.000% |
| ftsN_xy1 home | lab | MIP | 1.28 | 1.31 | 1.021 | same | 0.000% |
| ftsN_xy1 home | imglab50 | EA | 1.62 | 1.68 | 1.042 | same | 0.000% |
| ftsN_xy1 home | imglab50 | MIP | 3.63 | 3.45 | 0.949 | same | 0.000% |
| ftsN_xy1 home | imglab50 | mean | 3.57 | 3.60 | 1.008 | same | 0.000% |
| ftsN_xy1 zoom | imglab | EA | 5.90 | 6.02 | 1.020 | same | 0.000% |
| ftsN_xy1 zoom | imglab | MIP | 17.37 | 17.23 | 0.992 | same | 0.000% |
| ftsN_xy1 zoom | imglab | mean | 17.74 | 17.14 | 0.966 | same | 0.000% |
| ftsN_xy1 zoom | lab | MIP | 4.98 | 4.47 | 0.898 | same | 0.000% |
| ftsN_xy1 zoom | imglab50 | EA | 6.06 | 6.13 | 1.011 | same | 0.000% |
| ftsN_xy1 zoom | imglab50 | MIP | 16.98 | 17.09 | 1.006 | same | 0.000% |
| ftsN_xy1 zoom | imglab50 | mean | 16.91 | 17.82 | 1.054 | same | 0.000% |
| ftsN_xy1 top | imglab | EA | 1.89 | 1.97 | 1.043 | same | 0.000% |
| ftsN_xy1 top | imglab | MIP | 3.45 | 3.64 | 1.055 | same | 0.000% |
| ftsN_xy1 top | imglab | mean | 3.52 | 3.63 | 1.033 | same | 0.000% |
| ftsN_xy1 top | lab | MIP | 1.46 | 1.53 | 1.050 | same | 0.000% |
| ftsN_xy1 top | imglab50 | EA | 1.90 | 1.94 | 1.018 | same | 0.000% |
| ftsN_xy1 top | imglab50 | MIP | 3.58 | 3.55 | 0.991 | same | 0.000% |
| ftsN_xy1 top | imglab50 | mean | 3.48 | 3.57 | 1.023 | same | 0.000% |
| cells3d home | imglab | EA | 0.92 | 0.95 | 1.032 | same | 0.000% |
| cells3d home | imglab | MIP | 1.25 | 1.21 | 0.966 | same | 0.000% |
| cells3d home | imglab | mean | 1.26 | 1.22 | 0.965 | same | 0.000% |
| cells3d home | lab | MIP | 0.41 | 0.42 | 1.010 | same | 0.000% |
| cells3d home | imglab50 | EA | 0.91 | 0.92 | 1.003 | same | 0.000% |
| cells3d home | imglab50 | MIP | 1.25 | 1.26 | 1.006 | same | 0.000% |
| cells3d home | imglab50 | mean | 1.26 | 1.22 | 0.972 | same | 0.000% |
| cells3d zoom | imglab | EA | 4.96 | 4.95 | 0.998 | same | 0.000% |
| cells3d zoom | imglab | MIP | 6.46 | 6.62 | 1.026 | same | 0.000% |
| cells3d zoom | imglab | mean | 6.48 | 6.61 | 1.019 | same | 0.000% |
| cells3d zoom | lab | MIP | 2.03 | 2.06 | 1.014 | same | 0.000% |
| cells3d zoom | imglab50 | EA | 5.00 | 4.91 | 0.982 | same | 0.000% |
| cells3d zoom | imglab50 | MIP | 6.57 | 6.72 | 1.023 | same | 0.000% |
| cells3d zoom | imglab50 | mean | 6.48 | 6.84 | 1.054 | same | 0.000% |
| cells3d top | imglab | EA | 1.32 | 1.36 | 1.031 | same | 0.000% |
| cells3d top | imglab | MIP | 1.27 | 1.29 | 1.018 | same | 0.000% |
| cells3d top | imglab | mean | 1.38 | 1.36 | 0.984 | same | 0.000% |
| cells3d top | lab | MIP | 0.47 | 0.50 | 1.061 | same | 0.000% |
| cells3d top | imglab50 | EA | 1.42 | 1.46 | 1.024 | same | 0.000% |
| cells3d top | imglab50 | MIP | 1.32 | 1.31 | 0.997 | same | 0.000% |
| cells3d top | imglab50 | mean | 1.30 | 1.36 | 1.046 | same | 0.000% |

### idbuf

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | imglab | EA | 1.17 | 1.16 | 0.994 | same | 0.000% |
| dnaA_xy1 home | imglab | MIP | 2.26 | 2.26 | 1.002 | same | 0.000% |
| dnaA_xy1 home | imglab | mean | 2.25 | 2.25 | 1.000 | same | 0.000% |
| dnaA_xy1 home | lab | MIP | 0.91 | 0.90 | 0.992 | same | 0.000% |
| dnaA_xy1 home | imglab50 | EA | 1.17 | 1.17 | 0.997 | same | 0.000% |
| dnaA_xy1 home | imglab50 | MIP | 2.26 | 2.24 | 0.991 | same | 0.000% |
| dnaA_xy1 home | imglab50 | mean | 2.24 | 2.25 | 1.003 | same | 0.000% |
| dnaA_xy1 zoom | imglab | EA | 5.97 | 5.97 | 0.999 | same | 0.000% |
| dnaA_xy1 zoom | imglab | MIP | 14.03 | 13.61 | 0.970 | same | 0.000% |
| dnaA_xy1 zoom | imglab | mean | 14.46 | 14.41 | 0.996 | same | 0.000% |
| dnaA_xy1 zoom | lab | MIP | 5.26 | 5.27 | 1.002 | same | 0.000% |
| dnaA_xy1 zoom | imglab50 | EA | 7.02 | 6.99 | 0.996 | same | 0.000% |
| dnaA_xy1 zoom | imglab50 | MIP | 16.47 | 15.81 | 0.960 | same | 0.000% |
| dnaA_xy1 zoom | imglab50 | mean | 17.60 | 17.31 | 0.983 | same | 0.000% |
| dnaA_xy1 top | imglab | EA | 1.85 | 1.84 | 0.996 | same | 0.000% |
| dnaA_xy1 top | imglab | MIP | 3.06 | 3.09 | 1.010 | same | 0.000% |
| dnaA_xy1 top | imglab | mean | 3.71 | 3.46 | 0.931 | same | 0.000% |
| dnaA_xy1 top | lab | MIP | 1.09 | 1.08 | 0.989 | same | 0.000% |
| dnaA_xy1 top | imglab50 | EA | 1.83 | 1.84 | 1.005 | same | 0.000% |
| dnaA_xy1 top | imglab50 | MIP | 3.17 | 3.22 | 1.018 | same | 0.000% |
| dnaA_xy1 top | imglab50 | mean | 2.98 | 2.98 | 1.000 | same | 0.000% |
| 5I home | imglab | EA | 1.63 | 1.66 | 1.017 | same | 0.000% |
| 5I home | imglab | MIP | 3.71 | 3.63 | 0.978 | same | 0.000% |
| 5I home | imglab | mean | 3.46 | 3.40 | 0.980 | same | 0.000% |
| 5I home | lab | MIP | 1.26 | 1.27 | 1.008 | same | 0.000% |
| 5I home | imglab50 | EA | 1.61 | 1.69 | 1.045 | same | 0.000% |
| 5I home | imglab50 | MIP | 3.37 | 3.36 | 0.997 | same | 0.000% |
| 5I home | imglab50 | mean | 3.31 | 3.29 | 0.993 | same | 0.000% |
| 5I zoom | imglab | EA | 9.30 | 8.88 | 0.954 | same | 0.000% |
| 5I zoom | imglab | MIP | 21.75 | 18.92 | 0.870 | same | 0.000% |
| 5I zoom | imglab | mean | 19.11 | 20.12 | 1.053 | same | 0.000% |
| 5I zoom | lab | MIP | 7.24 | 7.12 | 0.983 | same | 0.000% |
| 5I zoom | imglab50 | EA | 8.73 | 8.72 | 0.999 | same | 0.000% |
| 5I zoom | imglab50 | MIP | 18.34 | 19.70 | 1.074 | same | 0.000% |
| 5I zoom | imglab50 | mean | 18.34 | 18.48 | 1.008 | same | 0.000% |
| 5I top | imglab | EA | 1.72 | 1.61 | 0.935 | same | 0.000% |
| 5I top | imglab | MIP | 3.01 | 3.09 | 1.025 | same | 0.000% |
| 5I top | imglab | mean | 3.00 | 3.05 | 1.018 | same | 0.000% |
| 5I top | lab | MIP | 1.10 | 1.10 | 0.997 | same | 0.000% |
| 5I top | imglab50 | EA | 1.27 | 1.26 | 0.997 | same | 0.000% |
| 5I top | imglab50 | MIP | 2.61 | 2.77 | 1.064 | same | 0.000% |
| 5I top | imglab50 | mean | 3.23 | 3.25 | 1.004 | same | 0.000% |
| ftsN_xy1 home | imglab | EA | 1.61 | 1.63 | 1.015 | same | 0.000% |
| ftsN_xy1 home | imglab | MIP | 3.66 | 3.65 | 0.998 | same | 0.000% |
| ftsN_xy1 home | imglab | mean | 3.61 | 3.56 | 0.984 | same | 0.000% |
| ftsN_xy1 home | lab | MIP | 1.28 | 1.31 | 1.019 | same | 0.000% |
| ftsN_xy1 home | imglab50 | EA | 1.62 | 1.64 | 1.016 | same | 0.000% |
| ftsN_xy1 home | imglab50 | MIP | 3.63 | 3.47 | 0.956 | same | 0.000% |
| ftsN_xy1 home | imglab50 | mean | 3.57 | 3.55 | 0.994 | same | 0.000% |
| ftsN_xy1 zoom | imglab | EA | 5.90 | 5.84 | 0.990 | same | 0.000% |
| ftsN_xy1 zoom | imglab | MIP | 17.37 | 17.13 | 0.987 | same | 0.000% |
| ftsN_xy1 zoom | imglab | mean | 17.74 | 17.05 | 0.961 | same | 0.000% |
| ftsN_xy1 zoom | lab | MIP | 4.98 | 5.03 | 1.011 | same | 0.000% |
| ftsN_xy1 zoom | imglab50 | EA | 6.06 | 5.96 | 0.983 | same | 0.000% |
| ftsN_xy1 zoom | imglab50 | MIP | 16.98 | 17.12 | 1.008 | same | 0.000% |
| ftsN_xy1 zoom | imglab50 | mean | 16.91 | 16.94 | 1.002 | same | 0.000% |
| ftsN_xy1 top | imglab | EA | 1.89 | 1.90 | 1.004 | same | 0.000% |
| ftsN_xy1 top | imglab | MIP | 3.45 | 3.61 | 1.046 | same | 0.000% |
| ftsN_xy1 top | imglab | mean | 3.52 | 3.50 | 0.995 | same | 0.000% |
| ftsN_xy1 top | lab | MIP | 1.46 | 1.44 | 0.987 | same | 0.000% |
| ftsN_xy1 top | imglab50 | EA | 1.90 | 2.03 | 1.064 | same | 0.000% |
| ftsN_xy1 top | imglab50 | MIP | 3.58 | 3.47 | 0.970 | same | 0.000% |
| ftsN_xy1 top | imglab50 | mean | 3.48 | 3.58 | 1.029 | same | 0.000% |
| cells3d home | imglab | EA | 0.92 | 0.94 | 1.019 | same | 0.000% |
| cells3d home | imglab | MIP | 1.25 | 1.20 | 0.959 | same | 0.000% |
| cells3d home | imglab | mean | 1.26 | 1.21 | 0.959 | same | 0.000% |
| cells3d home | lab | MIP | 0.41 | 0.41 | 0.998 | same | 0.000% |
| cells3d home | imglab50 | EA | 0.91 | 0.89 | 0.976 | same | 0.000% |
| cells3d home | imglab50 | MIP | 1.25 | 1.32 | 1.051 | same | 0.000% |
| cells3d home | imglab50 | mean | 1.26 | 1.23 | 0.980 | same | 0.000% |
| cells3d zoom | imglab | EA | 4.96 | 4.89 | 0.986 | same | 0.000% |
| cells3d zoom | imglab | MIP | 6.46 | 6.44 | 0.997 | same | 0.000% |
| cells3d zoom | imglab | mean | 6.48 | 6.54 | 1.010 | same | 0.000% |
| cells3d zoom | lab | MIP | 2.03 | 2.02 | 0.993 | same | 0.000% |
| cells3d zoom | imglab50 | EA | 5.00 | 4.90 | 0.982 | same | 0.000% |
| cells3d zoom | imglab50 | MIP | 6.57 | 6.48 | 0.987 | same | 0.000% |
| cells3d zoom | imglab50 | mean | 6.48 | 6.54 | 1.009 | same | 0.000% |
| cells3d top | imglab | EA | 1.32 | 1.33 | 1.005 | same | 0.000% |
| cells3d top | imglab | MIP | 1.27 | 1.31 | 1.034 | same | 0.000% |
| cells3d top | imglab | mean | 1.38 | 1.27 | 0.915 | same | 0.000% |
| cells3d top | lab | MIP | 0.47 | 0.47 | 1.004 | same | 0.000% |
| cells3d top | imglab50 | EA | 1.42 | 1.36 | 0.957 | same | 0.000% |
| cells3d top | imglab50 | MIP | 1.32 | 1.32 | 1.000 | same | 0.000% |
| cells3d top | imglab50 | mean | 1.30 | 1.31 | 1.010 | same | 0.000% |

### override

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.28 | 0.935 | faster | 0.000% |
| dnaA_xy1 home | img | MIP | 1.35 | 1.02 | 0.752 | faster | 0.000% |
| dnaA_xy1 home | img | mean | 1.37 | 1.02 | 0.747 | faster | 0.000% |
| dnaA_xy1 home | imglab | EA | 1.17 | 1.15 | 0.988 | same | 0.000% |
| dnaA_xy1 home | imglab | MIP | 2.26 | 1.91 | 0.845 | faster | 0.000% |
| dnaA_xy1 home | imglab | mean | 2.25 | 1.90 | 0.845 | faster | 0.000% |
| dnaA_xy1 home | lab | MIP | 0.91 | 0.93 | 1.017 | same | 0.000% |
| dnaA_xy1 home | imglab50 | EA | 1.17 | 1.16 | 0.989 | same | 0.000% |
| dnaA_xy1 home | imglab50 | MIP | 2.26 | 1.90 | 0.840 | faster | 0.000% |
| dnaA_xy1 home | imglab50 | mean | 2.24 | 1.90 | 0.847 | faster | 0.000% |
| dnaA_xy1 zoom | img | EA | 1.19 | 1.10 | 0.921 | faster | 0.000% |
| dnaA_xy1 zoom | img | MIP | 7.95 | 5.78 | 0.727 | faster | 0.000% |
| dnaA_xy1 zoom | img | mean | 8.34 | 6.03 | 0.723 | faster | 0.000% |
| dnaA_xy1 zoom | imglab | EA | 5.97 | 5.87 | 0.983 | same | 0.000% |
| dnaA_xy1 zoom | imglab | MIP | 14.03 | 11.45 | 0.817 | faster | 0.000% |
| dnaA_xy1 zoom | imglab | mean | 14.46 | 11.91 | 0.823 | faster | 0.000% |
| dnaA_xy1 zoom | lab | MIP | 5.26 | 5.31 | 1.009 | same | 0.000% |
| dnaA_xy1 zoom | imglab50 | EA | 7.02 | 6.85 | 0.976 | same | 0.000% |
| dnaA_xy1 zoom | imglab50 | MIP | 16.47 | 13.09 | 0.795 | faster | 0.000% |
| dnaA_xy1 zoom | imglab50 | mean | 17.60 | 14.37 | 0.816 | faster | 0.000% |
| dnaA_xy1 top | img | EA | 0.55 | 0.51 | 0.925 | same | 0.000% |
| dnaA_xy1 top | img | MIP | 1.92 | 1.47 | 0.765 | same | 0.000% |
| dnaA_xy1 top | img | mean | 1.97 | 1.61 | 0.816 | same | 0.000% |
| dnaA_xy1 top | imglab | EA | 1.85 | 1.88 | 1.018 | same | 0.000% |
| dnaA_xy1 top | imglab | MIP | 3.06 | 2.57 | 0.838 | faster | 0.000% |
| dnaA_xy1 top | imglab | mean | 3.71 | 3.01 | 0.812 | same | 0.000% |
| dnaA_xy1 top | lab | MIP | 1.09 | 1.10 | 1.012 | same | 0.000% |
| dnaA_xy1 top | imglab50 | EA | 1.83 | 1.79 | 0.974 | same | 0.000% |
| dnaA_xy1 top | imglab50 | MIP | 3.17 | 2.63 | 0.831 | same | 0.000% |
| dnaA_xy1 top | imglab50 | mean | 2.98 | 2.50 | 0.840 | faster | 0.000% |
| 5I home | img | EA | 0.27 | 0.00 | 0.000 | faster | 0.000% |
| 5I home | img | MIP | 2.02 | 0.00 | 0.000 | faster | 0.000% |
| 5I home | img | mean | 1.97 | 0.00 | 0.000 | faster | 0.000% |
| 5I home | imglab | EA | 1.63 | 0.00 | 0.000 | faster | 0.000% |
| 5I home | imglab | MIP | 3.71 | 0.00 | 0.000 | faster | 0.000% |
| 5I home | imglab | mean | 3.46 | 0.00 | 0.000 | faster | 0.000% |
| 5I home | lab | MIP | 1.26 | 0.00 | 0.000 | faster | 0.000% |
| 5I home | imglab50 | EA | 1.61 | 0.00 | 0.000 | faster | 0.000% |
| 5I home | imglab50 | MIP | 3.37 | 0.00 | 0.000 | faster | 0.000% |
| 5I home | imglab50 | mean | 3.31 | 0.00 | 0.000 | faster | 0.000% |
| 5I zoom | img | EA | 1.78 | 0.00 | 0.000 | faster | 0.000% |
| 5I zoom | img | MIP | 11.51 | 0.00 | 0.000 | faster | 0.000% |
| 5I zoom | img | mean | 12.49 | 0.00 | 0.000 | faster | 0.000% |
| 5I zoom | imglab | EA | 9.30 | 0.00 | 0.000 | faster | 0.000% |
| 5I zoom | imglab | MIP | 21.75 | 0.00 | 0.000 | faster | 0.000% |
| 5I zoom | imglab | mean | 19.11 | 0.00 | 0.000 | faster | 0.000% |
| 5I zoom | lab | MIP | 7.24 | 0.00 | 0.000 | faster | 0.000% |
| 5I zoom | imglab50 | EA | 8.73 | 0.00 | 0.000 | faster | 0.000% |
| 5I zoom | imglab50 | MIP | 18.34 | 0.00 | 0.000 | faster | 0.000% |
| 5I zoom | imglab50 | mean | 18.34 | 0.00 | 0.000 | faster | 0.000% |
| 5I top | img | EA | 0.41 | 0.00 | 0.000 | faster | 0.000% |
| 5I top | img | MIP | 1.84 | 0.00 | 0.000 | faster | 0.000% |
| 5I top | img | mean | 1.82 | 0.00 | 0.000 | faster | 0.000% |
| 5I top | imglab | EA | 1.72 | 0.00 | 0.000 | faster | 0.000% |
| 5I top | imglab | MIP | 3.01 | 0.00 | 0.000 | faster | 0.000% |
| 5I top | imglab | mean | 3.00 | 0.00 | 0.000 | faster | 0.000% |
| 5I top | lab | MIP | 1.10 | 0.00 | 0.000 | faster | 0.000% |
| 5I top | imglab50 | EA | 1.27 | 0.00 | 0.000 | faster | 0.000% |
| 5I top | imglab50 | MIP | 2.61 | 0.00 | 0.000 | faster | 0.000% |
| 5I top | imglab50 | mean | 3.23 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 home | img | EA | 0.27 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 home | img | MIP | 1.67 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 home | img | mean | 1.70 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 home | imglab | EA | 1.61 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 home | imglab | MIP | 3.66 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 home | imglab | mean | 3.61 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 home | lab | MIP | 1.28 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | EA | 1.62 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | MIP | 3.63 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 home | imglab50 | mean | 3.57 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 zoom | img | EA | 1.44 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 zoom | img | MIP | 13.89 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 zoom | img | mean | 12.42 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 zoom | imglab | EA | 5.90 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 zoom | imglab | MIP | 17.37 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 zoom | imglab | mean | 17.74 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 zoom | lab | MIP | 4.98 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | EA | 6.06 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | MIP | 16.98 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 zoom | imglab50 | mean | 16.91 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 top | img | EA | 0.38 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 top | img | MIP | 1.88 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 top | img | mean | 1.95 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 top | imglab | EA | 1.89 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 top | imglab | MIP | 3.45 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 top | imglab | mean | 3.52 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 top | lab | MIP | 1.46 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 top | imglab50 | EA | 1.90 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 top | imglab50 | MIP | 3.58 | 0.00 | 0.000 | faster | 0.000% |
| ftsN_xy1 top | imglab50 | mean | 3.48 | 0.00 | 0.000 | faster | 0.000% |
| cells3d home | img | EA | 0.52 | 0.00 | 0.000 | faster | 0.000% |
| cells3d home | img | MIP | 0.84 | 0.00 | 0.000 | faster | 0.000% |
| cells3d home | img | mean | 0.83 | 0.00 | 0.000 | faster | 0.000% |
| cells3d home | imglab | EA | 0.92 | 0.00 | 0.000 | faster | 0.000% |
| cells3d home | imglab | MIP | 1.25 | 0.00 | 0.000 | faster | 0.000% |
| cells3d home | imglab | mean | 1.26 | 0.00 | 0.000 | faster | 0.000% |
| cells3d home | lab | MIP | 0.41 | 0.00 | 0.000 | faster | 0.000% |
| cells3d home | imglab50 | EA | 0.91 | 0.00 | 0.000 | faster | 0.000% |
| cells3d home | imglab50 | MIP | 1.25 | 0.00 | 0.000 | faster | 0.000% |
| cells3d home | imglab50 | mean | 1.26 | 0.00 | 0.000 | faster | 0.000% |
| cells3d zoom | img | EA | 2.81 | 0.00 | 0.000 | faster | 0.000% |
| cells3d zoom | img | MIP | 4.46 | 0.00 | 0.000 | faster | 0.000% |
| cells3d zoom | img | mean | 4.75 | 0.00 | 0.000 | faster | 0.000% |
| cells3d zoom | imglab | EA | 4.96 | 0.00 | 0.000 | faster | 0.000% |
| cells3d zoom | imglab | MIP | 6.46 | 0.00 | 0.000 | faster | 0.000% |
| cells3d zoom | imglab | mean | 6.48 | 0.00 | 0.000 | faster | 0.000% |
| cells3d zoom | lab | MIP | 2.03 | 0.00 | 0.000 | faster | 0.000% |
| cells3d zoom | imglab50 | EA | 5.00 | 0.00 | 0.000 | faster | 0.000% |
| cells3d zoom | imglab50 | MIP | 6.57 | 0.00 | 0.000 | faster | 0.000% |
| cells3d zoom | imglab50 | mean | 6.48 | 0.00 | 0.000 | faster | 0.000% |
| cells3d top | img | EA | 0.85 | 0.00 | 0.000 | faster | 0.000% |
| cells3d top | img | MIP | 0.82 | 0.00 | 0.000 | faster | 0.000% |
| cells3d top | img | mean | 0.79 | 0.00 | 0.000 | faster | 0.000% |
| cells3d top | imglab | EA | 1.32 | 0.00 | 0.000 | faster | 0.000% |
| cells3d top | imglab | MIP | 1.27 | 0.00 | 0.000 | faster | 0.000% |
| cells3d top | imglab | mean | 1.38 | 0.00 | 0.000 | faster | 0.000% |
| cells3d top | lab | MIP | 0.47 | 0.00 | 0.000 | faster | 0.000% |
| cells3d top | imglab50 | EA | 1.42 | 0.00 | 0.000 | faster | 0.000% |
| cells3d top | imglab50 | MIP | 1.32 | 0.00 | 0.000 | faster | 0.000% |
| cells3d top | imglab50 | mean | 1.30 | 0.00 | 0.000 | faster | 0.000% |

## Render: 2 scenarios, 2048x1280, 20 frames x 4 rounds per variant

| Variant | Scenarios | GPU time vs shipped (geo mean) | Best | Worst | Faster | Same | Slower | Max pixel diff |
|---|---|---|---|---|---|---|---|---|
| skip8 | 2 | 1.361 | 1.166 | 1.588 | 0 | 0 | 2 | 0.037 |
| override | 2 | 0.837 | 0.750 | 0.935 | 2 | 0 | 0 | 0.000 |
| combo8 | 2 | 1.100 | 0.909 | 1.333 | 1 | 0 | 1 | 0.037 |
| combo16 | 2 | 1.083 | 0.915 | 1.281 | 1 | 0 | 1 | 0.057 |
| skipx8 | 2 | 1.091 | 0.950 | 1.253 | 1 | 0 | 1 | 0.037 |
| combo8x | 2 | 0.909 | 0.731 | 1.132 | 1 | 0 | 1 | 0.037 |
| combo16x | 2 | 0.907 | 0.755 | 1.089 | 1 | 0 | 1 | 0.057 |

### skip8

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.48 | 1.588 | slower | 0.000% |
| dnaA_xy1 home | img | MIP | 1.36 | 1.58 | 1.166 | slower | 0.000% |

### override

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.28 | 0.935 | faster | 0.000% |
| dnaA_xy1 home | img | MIP | 1.36 | 1.02 | 0.750 | faster | 0.000% |

### combo8

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.40 | 1.333 | slower | 0.000% |
| dnaA_xy1 home | img | MIP | 1.36 | 1.23 | 0.909 | faster | 0.000% |

### combo16

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.39 | 1.281 | slower | 0.000% |
| dnaA_xy1 home | img | MIP | 1.36 | 1.24 | 0.915 | faster | 0.000% |

### skipx8

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.38 | 1.253 | slower | 0.000% |
| dnaA_xy1 home | img | MIP | 1.36 | 1.29 | 0.950 | faster | 0.000% |

### combo8x

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.34 | 1.132 | slower | 0.000% |
| dnaA_xy1 home | img | MIP | 1.36 | 0.99 | 0.731 | faster | 0.000% |

### combo16x

| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |
|---|---|---|---|---|---|---|---|
| dnaA_xy1 home | img | EA | 0.30 | 0.33 | 1.089 | slower | 0.000% |
| dnaA_xy1 home | img | MIP | 1.36 | 1.03 | 0.755 | faster | 0.000% |

## bricks

{"ds": "dnaA_xy1", "B": 4, "build_ms": 24.40000000037253, "lab_brick_occupancy": 0.1263443050350334, "dims": [76, 76, 34]}
{"ds": "dnaA_xy1", "B": 8, "build_ms": 15.199999999254942, "lab_brick_occupancy": 0.1514583672804302, "dims": [38, 38, 17]}
{"ds": "dnaA_xy1", "B": 16, "build_ms": 13.900000000372529, "lab_brick_occupancy": 0.21052631578947367, "dims": [19, 19, 9]}
{"ds": "5I", "B": 4, "build_ms": 43.20000000111759, "lab_brick_occupancy": 0.07707223623960377, "dims": [82, 87, 36]}
{"ds": "5I", "B": 8, "build_ms": 29.40000000037253, "lab_brick_occupancy": 0.08890736634639074, "dims": [41, 44, 18]}
{"ds": "5I", "B": 16, "build_ms": 25.59999999962747, "lab_brick_occupancy": 0.11471861471861472, "dims": [21, 22, 9]}
{"ds": "ftsN_xy1", "B": 4, "build_ms": 120.90000000037253, "lab_brick_occupancy": 0.06616628239264402, "dims": [59, 151, 61]}
{"ds": "ftsN_xy1", "B": 8, "build_ms": 49.19999999925494, "lab_brick_occupancy": 0.08173457838143747, "dims": [30, 76, 31]}
{"ds": "ftsN_xy1", "B": 16, "build_ms": 44, "lab_brick_occupancy": 0.11885964912280701, "dims": [15, 38, 16]}
{"ds": "cells3d", "B": 4, "build_ms": 8.299999998882413, "lab_brick_occupancy": 0.243603515625, "dims": [64, 64, 15]}
{"ds": "cells3d", "B": 8, "build_ms": 6.099999999627471, "lab_brick_occupancy": 0.3316650390625, "dims": [32, 32, 8]}
{"ds": "cells3d", "B": 16, "build_ms": 5.699999999254942, "lab_brick_occupancy": 0.5263671875, "dims": [16, 16, 4]}
{"ds": "cells3d", "B": 4, "build_ms": 10.199999999254942, "lab_brick_occupancy": 0.243603515625, "dims": [64, 64, 15]}
{"ds": "cells3d", "B": 8, "build_ms": 5.099999999627471, "lab_brick_occupancy": 0.3316650390625, "dims": [32, 32, 8]}
{"ds": "cells3d", "B": 16, "build_ms": 4.599999999627471, "lab_brick_occupancy": 0.5263671875, "dims": [16, 16, 4]}
{"ds": "dnaA_xy1", "B": 4, "build_ms": 24.90000000037253, "lab_brick_occupancy": 0.1263443050350334, "dims": [76, 76, 34]}
{"ds": "dnaA_xy1", "B": 8, "build_ms": 15.800000000745058, "lab_brick_occupancy": 0.1514583672804302, "dims": [38, 38, 17]}
{"ds": "dnaA_xy1", "B": 16, "build_ms": 14.099999999627471, "lab_brick_occupancy": 0.21052631578947367, "dims": [19, 19, 9]}
{"ds": "dnaA_xy1", "B": 4, "build_ms": 24.40000000037253, "lab_brick_occupancy": 0.1263443050350334, "dims": [76, 76, 34]}
{"ds": "dnaA_xy1", "B": 8, "build_ms": 15.300000000745058, "lab_brick_occupancy": 0.1514583672804302, "dims": [38, 38, 17]}
{"ds": "dnaA_xy1", "B": 16, "build_ms": 13.699999999254942, "lab_brick_occupancy": 0.21052631578947367, "dims": [19, 19, 9]}

