"""Carga + limpieza + agregación diaria -> data/{static,daily,trips}.parquet y quality.json."""
import json
from pathlib import Path

import duckdb

from src.config import CFG

RAW = Path("Datasets")
OUT = Path("data")
SF1 = RAW / "Static/StaticInformation_FailedVins.csv"
SF2 = RAW / "Static/StaticInformation_FailedVins_v2_20260924_144239.csv"
SN2 = RAW / "Static/StaticInformation_NotFailedVins_v2_20260924_144209.csv"
TRIPS = {"failed": RAW / "TripSummary/TripSummary_Failed_SelectionVins_vehiclecode_v2 (2).csv",
         "healthy": RAW / "TripSummary/TripSummary_NotFailed_SelectionVins.csv"}
DYN = {"failed": RAW / "Dynamic/DynamicInformation_Failed_SelectionVins_v2.csv",
       "healthy": RAW / "Dynamic/DynamicInformation_NotFailed_SelectionVins.csv"}
# Fallados que solo están en v1 (sin fecha de falla confiable): se usan únicamente en el pre-entrenamiento auto-supervisado
TRIPS_V1 = {"failed_v1": RAW / "TripSummary/TripSummary_Failed_SelectionVins.csv"}
DYN_V1 = {"failed_v1": RAW / "Dynamic/DynamicInformation_Failed_SelectionVins.csv"}
LOCAL = "INTERVAL 4 HOUR"  # hora local ≈ UTC-4 fijo


# Estados del DPF (renombrado "Air Filter" en el dataset anonimizado)
OVER = "('Air Filter Over Limit','Air Filter Overloaded','Air Filter At Limit')"


def csv(p):
    return f"read_csv('{p}', header=true, all_varchar=true)"


def num(col, lo, hi):
    """Casteo numérico; fuera de rango físico -> NULL."""
    return f"case when try_cast({col} as double) between {lo} and {hi} then try_cast({col} as double) end"


def build(v1_only=False):
    """v1_only: solo fallados exclusivos de v1 -> data/daily_v1.parquet (pre-entrenamiento)."""
    trips, dyn = (TRIPS_V1, DYN_V1) if v1_only else (TRIPS, DYN)
    OUT.mkdir(exist_ok=True)
    cfg = CFG["duckdb"]  # límites de recursos: src/config.py / config.local.json
    c = duckdb.connect(config=cfg)
    print("DuckDB:", cfg)
    q = {}

    # ---------- static ----------
    c.execute(f"""create table sf as select distinct VehicleCode v, IdentificationDaysSinceProduction::int idp,
        try_cast(daysUntilSale as int) ds, ProductionDay::int pd, Engine, ModelSeries, SalesCountry_cd country, SalesCity city
        from {csv(SF2)}""")
    c.execute(f"""create table sn as select VehicleCode v, any_value(try_cast(daysUntilSale as int)) ds, any_value(ProductionDay::int) pd,
        any_value(Engine) Engine, any_value(ModelSeries) ModelSeries, any_value(SalesCountry_cd) country, any_value(SalesCity) city
        from {csv(SN2)} group by 1""")
    conflict = c.execute(f"""select count(*) from sn where v in (select v from sf union select VehicleCode from {csv(SF1)})""").fetchone()[0]
    c.execute(f"delete from sn where v in (select v from sf union select VehicleCode from {csv(SF1)})")
    q["healthy_dropped_label_conflict"] = conflict
    q["failed_vehicles_with_multiple_events"] = c.execute("select count(*) from (select v from sf group by 1 having count(*)>1)").fetchone()[0]

    # ---------- trips ----------
    for g, p in trips.items():
        c.execute(f"""create table t_{g} as select VehicleCode v,
            try_cast(TripDatetimeStart as timestamptz) ts0, try_cast(TripDatetimeEnd as timestamptz) ts1,
            try_cast(OdometerTripStart as double) o0, try_cast(OdometerTripEnd as double) o1,
            {num('FuelLvlStartPc', 0, 105)} f0, {num('FuelLvlEndPc', 0, 105)} f1,
            {num('EngineOilLifePCStart', 0, 100)} oil0, {num('EngineOilLifePCEnd', 0, 100)} oil1,
            {num('EngineTemperatureMin', -40, 130)} etmin, {num('EngineTemperatureMax', -40, 130)} etmax,
            {num('EngineTemperatureAvg', -40, 130)} etavg,
            AirFilterStart dpf_state0, AirFilterEnd dpf_state1,
            {num('AirRegenerationStart', 0, 100)} soot0, {num('AirRegenerationEnd', 0, 100)} soot1,
            {num('CoolantTemperatureStart', -40, 130)} cool0, {num('CoolantTemperatureEnd', -40, 130)} cool1,
            {num('AirTemperatureAvg', -40, 55)} air, {num('AirTemperatureMin', -40, 55)} airmin
            from {csv(p)}""")
    c.execute("create table t_raw as " + " union all ".join(f"select * from t_{g}" for g in trips))
    q["trips_raw"] = c.execute("select count(*) from t_raw").fetchone()[0]
    rules = {
        "trips_bad_timestamp": "ts0 is null or ts1 is null or ts1 < ts0",
        "trips_duration_gt_24h": "epoch(ts1-ts0) > 86400",
        "trips_odometer_negative_delta": "o1 < o0 or o0 is null",
        "trips_km_gt_1500": "o1 - o0 > 1500",
        "trips_speed_gt_200kmh": "(o1-o0) / greatest(epoch(ts1-ts0),1) * 3600 > 200 and o1-o0 > 2",
    }
    for k, cond in rules.items():
        q[k] = c.execute(f"select count(*) from t_raw where {cond}").fetchone()[0]
    # duplicados (v, ts0): se conserva uno con desempate determinista (si no, el resultado depende de los hilos)
    c.execute(f"""create table t as select *,
        o1-o0 km, epoch(ts1-ts0)/60 mins, ((ts0 at time zone 'UTC') - {LOCAL}) lts
        from t_raw where not ({' or '.join('(' + r + ')' for r in rules.values())})
        qualify row_number() over (partition by v, ts0 order by ts1, o0, o1, f0, f1, oil0, oil1, etmin, etmax, etavg,
            soot0, soot1, cool0, cool1, air, airmin, dpf_state0, dpf_state1) = 1
        order by v, ts0""")
    valid = c.execute(f"select count(*) from t_raw where not ({' or '.join('(' + r + ')' for r in rules.values())})").fetchone()[0]
    q["trips_dropped_invalid_total"] = q["trips_raw"] - valid
    q["trips_duplicates_dropped"] = valid - c.execute("select count(*) from t").fetchone()[0]
    q["trips_clean"] = c.execute("select count(*) from t").fetchone()[0]
    q["trip_null_rate"] = c.execute("select " + ",".join(
        f"avg(({x} is null)::int) {x}" for x in ["f0", "oil0", "etmax", "soot0", "cool0", "air"]) + " from t").df().round(4).iloc[0].to_dict()

    # ---------- production date (D0 estimado por cohorte) ----------
    for s in ["sf", "sn"]:
        c.execute(f"alter table {s} add column prod date")
        d0 = c.execute(f"""select median(epoch(ft)/86400 - pd) from (select v, min(ts0) ft from t group by 1) join {s} using(v)""").fetchone()[0]
        if d0 is None:  # modo v1_only: no hay viajes de esta cohorte
            continue
        disp = c.execute(f"""select quantile_cont(epoch(ft)/86400 - pd - {d0}, 0.95) - quantile_cont(epoch(ft)/86400 - pd - {d0}, 0.05)
            from (select v, min(ts0) ft from t group by 1) join {s} using(v)""").fetchone()[0]
        q[f"d0_{s}"] = round(d0, 2)
        q[f"d0_{s}_p5_p95_spread_days"] = round(disp, 2)
        c.execute(f"update {s} set prod = (to_timestamp(({d0}::double + pd)*86400))::date")
    c.execute("""create table static as
        select v, 1 failed, list(distinct (prod + idp*interval 1 day)::date order by (prod + idp*interval 1 day)::date) events,
               any_value(prod) prod, any_value(ds) ds, any_value(Engine) Engine, any_value(ModelSeries) ModelSeries,
               any_value(country) country, any_value(city) city from sf group by v
        union all
        select v, 0, []::date[], prod, ds, Engine, ModelSeries, country, city from sn""")

    # ---------- dynamic (señales ECU del DPF) ----------
    for g, p in dyn.items():
        c.execute(f"""create table d_{g} as select distinct VehicleCode v, try_cast(eventTimestamp as timestamptz) ts,
            {num('Acumulation', 0, 100)} acc, try_cast(OdometerValue as double) odo, Message msg, Regenerations is not null reg,
            case when try_cast(DistanceBetweenRegenerations as double) between 0 and 20000
                 then try_cast(DistanceBetweenRegenerations as double) end dbr
            from {csv(p)}""")
    c.execute("create table d as " + " union all ".join(f"select * from d_{g}" for g in dyn))
    q["dynamic_rows"] = c.execute("select count(*) from d").fetchone()[0]
    q["dynamic_negative_dbr_dropped"] = c.execute("select count(*) from (" + " union all ".join(
        f"select * from {csv(p)}" for p in dyn.values()) + ") where try_cast(DistanceBetweenRegenerations as double) < 0").fetchone()[0]

    # ---------- regeneraciones reconstruidas desde la señal de hollín ----------
    # caída de hollín >= 20 pts entre lecturas (episodios a > 6 h); la bandera Regenerations se corta
    c.execute("""create table dl as select v, ts, odo, lag(acc) over w - acc soot_drop from d
        window w as (partition by v order by ts, acc, odo, msg, reg, dbr)""")
    c.execute("""create table rg as select v, ts, case when odo - lag(odo) over w between 0 and 20000
            then odo - lag(odo) over w end dbr
        from (select *, lag(ts) over (partition by v order by ts, odo, soot_drop) pts from dl where soot_drop >= 20)
        where pts is null or epoch(ts - pts) > 6*3600 window w as (partition by v order by ts, odo, soot_drop)""")
    c.execute("create table rp as select v, ts from dl where soot_drop >= 10 and soot_drop < 20")
    q["regen_reconstructed_episodes"] = c.execute("select count(*) from rg").fetchone()[0]
    flag_end = c.execute("""select max(day) + 1 from (select ts::date as "day", count(*) n from d where reg group by 1)
        where n >= 3""").fetchone()[0]
    q["regen_flag_outage_from"] = str(flag_end)
    q["regen_flag_msgs_after_outage"] = c.execute(f"select count(*) from d where ts >= '{flag_end}'").fetchone()[0]

    # ---------- agregación diaria ----------
    c.execute(f"""create table td as select v, lts::date as "day",
        count(*) n_trips, sum(km) km, sum(mins) mins,
        avg((km < 5)::int) sh_short5, avg((km < 10)::int) sh_short10, avg((km < 2)::int) sh_micro,
        sum(km) / nullif(sum(mins)/60, 0) speed, avg((km / nullif(mins/60, 0) < 25)::int) sh_urban,
        avg((etmax < 70)::int) sh_never_warm, avg((cool0 < 30)::int) sh_cold_start,
        avg(etmax) etmax, avg(etavg) etavg, avg(cool1 - cool0) cool_rise,
        sum(greatest(f0 - f1, 0)) fuel_used, sum(greatest(f0 - f1, 0)) / nullif(sum(km), 0) * 100 fuel_per100,
        max(soot1) soot_max, avg(soot1) soot_mean, sum(soot1 - soot0) soot_delta,
        sum(greatest(soot1 - soot0, 0)) / nullif(sum(km), 0) soot_per_km,
        avg((dpf_state1 like 'Cleaning Automatically%')::int) sh_trip_end_in_regen,
        sum((dpf_state0 like 'Cleaning%' and dpf_state1 not like 'Cleaning%')::int) regen_completed_in_trip,
        avg((dpf_state1 = 'Air Filter Full')::int) sh_end_full, avg((dpf_state1 in {OVER})::int) sh_end_over,
        min(oil1) oil_min, sum(greatest(oil0 - oil1, 0)) oil_drop, avg(air) air, min(airmin) air_min,
        avg((hour(lts) between 22 and 23 or hour(lts) between 0 and 5)::int) sh_night,
        avg((mins > 5 and km < 1)::int) sh_idle
        from t group by 1, 2""")
    c.execute(f"""create table dd as select v, ((ts at time zone 'UTC') - {LOCAL})::date as "day",
        count(*) n_msgs, avg(acc) acc_mean, max(acc) acc_max,
        avg((msg = 'Air Filter Full')::int) sh_full, avg((msg in {OVER})::int) sh_over,
        avg((msg = 'Air Filter Overloaded')::int) sh_overloaded,
        avg((msg like 'Cleaning Automatically%')::int) sh_regen_msg,
        sum((msg like 'Stopped Clean%')::int) n_regen_stopped, sum((msg like 'Cleanning Manually%')::int) n_manual_regen,
        from d group by 1, 2""")
    for tbl, cols in [("rg", "count(*) n_regen, min(dbr) dbr_min, avg(dbr) dbr_mean"), ("rp", "count(*) n_regen_partial")]:
        c.execute(f"""create table {tbl}d as select v, ((ts at time zone 'UTC') - {LOCAL})::date as "day", {cols}
            from {tbl} group by 1, 2""")
    c.execute("""create or replace table dd as select dd.*, coalesce(rgd.n_regen, 0) n_regen, rgd.dbr_min, rgd.dbr_mean,
        coalesce(rpd.n_regen_partial, 0) n_regen_partial
        from dd left join rgd using (v, "day") left join rpd using (v, "day")""")
    c.execute("""create table daily as select coalesce(td.v, dd.v) v, coalesce(td.day, dd.day) as "day", td.* exclude (v, day), dd.* exclude (v, day)
        from td full join dd on td.v = dd.v and td.day = dd.day""")
    q["vehicle_days"] = c.execute("select count(*) from daily").fetchone()[0]
    if v1_only:
        c.execute(f"""copy (select * from daily where v in (select VehicleCode from {csv(SF1)})
            and v not in (select v from static) order by v, day) to '{OUT}/daily_v1.parquet'""")
        return c.execute(f"select count(distinct v) vehicles, count(*) vehicle_days from '{OUT}/daily_v1.parquet'").df()
    q["vehicles"] = c.execute("select failed, count(*) from static group by 1").df().set_index("failed")["count_star()"].to_dict()

    c.execute(f"copy (select * from static order by v) to '{OUT}/static.parquet'")
    c.execute(f"copy (select * from daily where v in (select v from static) order by v, day) to '{OUT}/daily.parquet'")
    # viajes individuales (hora local): ventana de regeneración por vehículo en src/copilot.py
    c.execute(f"""copy (select v, lts, km, mins, soot0, soot1, dpf_state0, dpf_state1 from t
        where v in (select v from static) order by v, lts) to '{OUT}/trips.parquet'""")
    (OUT / "quality.json").write_text(json.dumps(q, indent=2, default=str))
    return q


if __name__ == "__main__":
    import sys
    if "v1" in sys.argv:
        print(build(v1_only=True))
        sys.exit()
    q = build()
    print(json.dumps(q, indent=2, default=str))
    assert q["d0_sf_p5_p95_spread_days"] < 2 and q["d0_sn_p5_p95_spread_days"] < 2, "anclaje de producción inconsistente"
