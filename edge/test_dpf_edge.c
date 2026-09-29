/* Arnés de paridad: stdin "vehículo,17 valores" -> stdout "vehículo,puntaje,alerta". */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dpf_edge.h"

int main(void) {
    char line[512], prev[512] = "";
    dpf_state_t s;
    dpf_reset(&s);
    while (fgets(line, sizeof line, stdin) != NULL) {
        dpf_day_t d;
        double v[DPF_N_BASE];
        char *p = strchr(line, ',');
        double raw;
        int i, alert;
        if (p == NULL) {
            continue;
        }
        *p = '\0';
        for (i = 0; i < DPF_N_BASE; i++) {
            v[i] = strtod(p + 1, &p);
        }
        d.n_trips = v[0]; d.n_msgs = v[1]; d.km = v[2]; d.mins = v[3]; d.n_regen = v[4]; d.n_regen_stopped = v[5];
        for (i = 0; i < 7; i++) d.trip_num[i] = v[6 + i];
        for (i = 0; i < 4; i++) d.msg_num[i] = v[13 + i];
        if (strcmp(line, prev) != 0) {
            dpf_reset(&s);
            memcpy(prev, line, strlen(line) + 1);
        }
        alert = dpf_update(&s, &d, &raw);
        printf("%s,%.17g,%d\n", line, raw, alert);
    }
    return 0;
}
