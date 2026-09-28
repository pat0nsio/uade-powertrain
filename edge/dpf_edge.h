/* DPF Health Copilot — alerta a bordo (ECU / módulo telemático).
 *
 * Memoria fija: el estado son medias móviles exponenciales (EMA) de 17 series diarias con 2 vidas medias, más dos
 * contadores desde la última regeneración. Sin memoria dinámica, sin recursión, C99.
 * El modelo (árboles, umbral, constantes de las EMA) está en dpf_edge_model.h, generado por
 * `python -m src.edge export` y separado del código como un set de calibración.
 */
#ifndef DPF_EDGE_H
#define DPF_EDGE_H

#include <stdint.h>

#define DPF_N_BASE 17      /* series diarias que se promedian */
#define DPF_N_HL 2         /* vidas medias: 7 y 30 días */
#define DPF_N_FEATURES 50  /* señales que ve el modelo */

/* Agregados de un día calendario (todo en 0 los días sin uso). Los numeradores son conteos del día. */
typedef struct {
    double n_trips;          /* viajes */
    double n_msgs;           /* mensajes de estado del DPF */
    double km;
    double mins;             /* minutos de manejo */
    double n_regen;          /* regeneraciones completas (caída de hollín >= 20 puntos) */
    double n_regen_stopped;  /* regeneraciones interrumpidas */
    /* viajes: < 5 km, < 2 km, urbanos (< 25 km/h), sin llegar a 70 °C, arranque en frío (< 30 °C),
       terminados durante una limpieza, ralentí (> 5 min y < 1 km) */
    double trip_num[7];
    /* mensajes: DPF sobre el límite (at limit / over limit / overloaded), lleno, sobrecargado; suma del % de hollín */
    double msg_num[4];
} dpf_day_t;

typedef struct {
    double ema[DPF_N_HL][DPF_N_BASE];
    double km_since_regen;
    double days_since_regen;
    uint8_t started;
    uint8_t regen_seen;
} dpf_state_t;

void dpf_reset(dpf_state_t *s);

/* Una vez por día calendario, también los días sin uso. Devuelve 1 si el vehículo entra en alerta.
   raw_score (opcional) recibe el puntaje del modelo en escala logit. */
int dpf_update(dpf_state_t *s, const dpf_day_t *d, double *raw_score);

void dpf_features(const dpf_state_t *s, double x[DPF_N_FEATURES]);
double dpf_score(const double x[DPF_N_FEATURES]);

#endif
