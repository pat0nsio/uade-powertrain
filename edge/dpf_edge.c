/* Réplica de src/edge.py (ema_features + LightGBM). */
#include <math.h>

#include "dpf_edge.h"
#include "dpf_edge_model.h"

enum { B_TRIPS = 0, B_MSGS, B_KM, B_HOURS, B_REGEN, B_STOPPED, B_TRIP_NUM, B_MSG_NUM = B_TRIP_NUM + 7 };
#define ZERO_EPS 1e-35f /* kZeroThreshold de LightGBM */

static double div_or_nan(double a, double b) { return (b == 0.0) ? NAN : a / b; }

/* EMA con adjust=False, misma aritmética que pandas.Series.ewm(...).mean() */
static void ema_step(double *w, double cur, double alpha) {
    const double old_wt = 1.0 - alpha;
    if (*w != cur) {
        *w = ((old_wt * *w) + (alpha * cur)) / (old_wt + alpha);
    }
}

void dpf_reset(dpf_state_t *s) {
    int h, i;
    for (h = 0; h < DPF_N_HL; h++) {
        for (i = 0; i < DPF_N_BASE; i++) {
            s->ema[h][i] = 0.0;
        }
    }
    s->km_since_regen = 0.0;
    s->days_since_regen = 0.0;
    s->started = 0U;
    s->regen_seen = 0U;
}

void dpf_features(const dpf_state_t *s, double x[DPF_N_FEATURES]) {
    int h, i, k = 0;
    for (h = 0; h < DPF_N_HL; h++) {
        const double *e = s->ema[h];
        for (i = 0; i < 7; i++) {
            x[k++] = div_or_nan(e[B_TRIP_NUM + i], e[B_TRIPS]);
        }
        for (i = 0; i < 4; i++) {
            x[k++] = div_or_nan(e[B_MSG_NUM + i], e[B_MSGS]);
        }
        x[k++] = div_or_nan(e[B_KM], e[B_TRIPS]);                       /* km por viaje */
        x[k++] = div_or_nan(e[B_KM], e[B_HOURS]);                       /* velocidad media */
        x[k++] = e[B_KM];                                               /* km por día */
        x[k++] = div_or_nan(e[B_REGEN], e[B_KM]) * 1000.0;              /* regeneraciones cada 1000 km */
        x[k++] = div_or_nan(e[B_STOPPED], e[B_REGEN] + e[B_STOPPED]);   /* fracción interrumpida */
    }
    for (i = 0; i < 16; i++) { /* tendencia: vida media 7 d - 30 d */
        x[k++] = x[i] - x[16 + i];
    }
    x[k++] = s->regen_seen ? s->km_since_regen : NAN;
    x[k] = s->regen_seen ? s->days_since_regen : NAN;
}

double dpf_score(const double x[DPF_N_FEATURES]) {
    double raw = 0.0;
    int t;
    for (t = 0; t < DPF_N_TREES; t++) {
        int32_t n = DPF_TREE_ROOT[t];
        while (n >= 0) { /* profundidad acotada por construcción (<= DPF_MAX_DEPTH) */
            double v = x[DPF_NODE_FEATURE[n]];
            const uint8_t mt = DPF_NODE_MISSING[n];
            int left;
            if (isnan(v) && mt != DPF_MISSING_NAN) {
                v = 0.0;
            }
            if ((mt == DPF_MISSING_ZERO && v >= -ZERO_EPS && v <= ZERO_EPS) || (mt == DPF_MISSING_NAN && isnan(v))) {
                left = DPF_NODE_DEFAULT_LEFT[n];
            } else {
                left = v <= DPF_NODE_THRESHOLD[n];
            }
            n = left ? DPF_NODE_LEFT[n] : DPF_NODE_RIGHT[n];
        }
        raw += DPF_LEAF_VALUE[~n];
    }
    return raw;
}

int dpf_update(dpf_state_t *s, const dpf_day_t *d, double *raw_score) {
    double b[DPF_N_BASE];
    double x[DPF_N_FEATURES];
    double raw;
    int h, i;
    b[B_TRIPS] = d->n_trips;
    b[B_MSGS] = d->n_msgs;
    b[B_KM] = d->km;
    b[B_HOURS] = d->mins / 60.0;
    b[B_REGEN] = d->n_regen;
    b[B_STOPPED] = d->n_regen_stopped;
    for (i = 0; i < 7; i++) {
        b[B_TRIP_NUM + i] = d->trip_num[i];
    }
    for (i = 0; i < 4; i++) {
        b[B_MSG_NUM + i] = d->msg_num[i];
    }
    for (h = 0; h < DPF_N_HL; h++) {
        for (i = 0; i < DPF_N_BASE; i++) {
            if (s->started) {
                ema_step(&s->ema[h][i], b[i], DPF_ALPHA[h]);
            } else {
                s->ema[h][i] = b[i];
            }
        }
    }
    s->started = 1U;
    if (d->n_regen > 0.0) {
        s->regen_seen = 1U;
        s->km_since_regen = d->km;
        s->days_since_regen = 0.0;
    } else if (s->regen_seen) {
        s->km_since_regen += d->km;
        s->days_since_regen += 1.0;
    }
    dpf_features(s, x);
    raw = dpf_score(x);
    if (raw_score != 0) {
        *raw_score = raw;
    }
    return raw >= DPF_THRESHOLD_RAW;
}
