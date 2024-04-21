-- 为方便查询，先选取AKI患者
select distinct stay_id from mimiciv_icu.icustays;
select stay_id from mimiciv_control.max_aki_stage;
-- 提取eGFR
/*
eGFR=186×(血清肌酐/88.4) ^−1.154 ×年龄 ^−0.203 ×(0.742 如果是女性)×(1.212 如果是黑人)

这里，血清肌酐的单位是微摩尔每升（μmol/L）（美国常用毫克/分升，需要转换），年龄以年为单位。
eGFR的不同阶段
肾脏疾病通常根据eGFR的数值被分为几个阶段，这有助于了解肾功能的减退程度：

正常或高值（G1）
eGFR ≥90 mL/min/1.73 m²
通常意味着肾功能正常或接近正常。
轻度下降（G2）
eGFR 60-89 mL/min/1.73 m²
可能仍被认为是在正常范围内，特别是在老年人中，因为肾功能可能会随年龄自然下降。
中度下降（G3a和G3b）
G3a: eGFR 45-59 mL/min/1.73 m²
G3b: eGFR 30-44 mL/min/1.73 m²
表明肾功能有中度损害。
重度下降（G4）
eGFR 15-29 mL/min/1.73 m²
表明肾功能严重受损。
肾衰竭（G5）
eGFR <15 mL/min/1.73 m²
通常需要肾脏替代治疗（如透析或肾脏移植）*/

-- -- 计算患者的eGFT
-- select subject_id, gender, anchor_age from mimiciv_hosp.patients;
-- select subject_id, race from mimiciv_hosp.admissions;
-- select subject_id, stay_id, charttime, value from mimiciv_icu.chartevents where itemid = 220615;

-- SELECT
--     p.subject_id,
--     p.gender,
--     p.anchor_age,
--     a.race,
--     ce.stay_id,
--     ce.charttime,
--     ce.value AS creatinine
-- FROM
--     mimiciv_hosp.patients p
-- JOIN
--     mimiciv_hosp.admissions a ON p.subject_id = a.subject_id
-- JOIN
--     mimiciv_icu.chartevents ce ON p.subject_id = ce.subject_id
-- WHERE
--     ce.itemid = 220615 -- 筛选血清肌酐记录

copy(SELECT
    p.subject_id,
    p.gender,
    p.anchor_age,
    a.race,
    ce.stay_id,
    ce.charttime,
    ce.value AS creatinine
FROM
    mimiciv_hosp.patients p
JOIN
    mimiciv_hosp.admissions a ON p.subject_id = a.subject_id
JOIN
    mimiciv_icu.chartevents ce ON p.subject_id = ce.subject_id
JOIN
    mimiciv_control.max_aki_stage mas ON ce.stay_id = mas.stay_id
WHERE
    ce.itemid = 220615 -- 筛选血清肌酐记录
ORDER BY
    ce.stay_id, ce.charttime)to 'E:/ICU_RRT/code/data/oral_egfr.csv' WITH CSV HEADER;


-- 提取RRT平均长度
select * from mimiciv_icu.procedureevents where itemid = 225802; -- CRRT
select * from mimiciv_icu.procedureevents where itemid = 225803;
select * from mimiciv_icu.procedureevents where itemid = 225809;
select * from mimiciv_icu.procedureevents where itemid = 225955;
select * from mimiciv_icu.procedureevents where itemid = 225802;
select * from mimiciv_icu.procedureevents where itemid = 225441; -- IHD

select * from mimiciv_derived.rrt;
copy(
select * from mimiciv_icu.procedureevents ce 
	JOIN
    mimiciv_control.max_aki_stage mas ON ce.stay_id = mas.stay_id
	where itemid = 225802 
) to 'E:/ICU_RRT/code/data/crrt.csv' WITH CSV HEADER;

copy(
select * from mimiciv_icu.procedureevents ce 
	JOIN
    mimiciv_control.max_aki_stage mas ON ce.stay_id = mas.stay_id
	where itemid = 225441 
) to 'E:/ICU_RRT/code/data/ihd.csv' WITH CSV HEADER;
/*想一想该怎么做
现在的轨迹记录可能是连续的，也有可能是不连续的。
找到患者的出入icu时间，以1小时为时间粒度，构建时间-stay_id的序列
查看是否有序列在其中，使用python做吧
stay_id, time_split, is_crrt, is_ihd, weight, originalrate 
*/
copy(
select * from mimiciv_icu.procedureevents ce 
	JOIN
    mimiciv_control.max_aki_stage mas ON ce.stay_id = mas.stay_id
	where itemid = 225441 
) to 'E:/ICU_RRT/code/data/ihd.csv' WITH CSV HEADER;

copy(
select ce.stay_id, intime, outtime from mimiciv_icu.icustays ce
		JOIN
    mimiciv_control.max_aki_stage mas ON ce.stay_id = mas.stay_id
) to 'E:/ICU_RRT/code/data/stay.csv' WITH CSV HEADER;

select stay_id, intime, outtime from mimiciv_icu.icustays;
WITH RECURSIVE hourly_slices AS (
  -- 初始化查询，获取每个住院时段的起始时间
  SELECT stay_id, intime AS start_time, 
         LEAST(DATE_ADD(intime, INTERVAL 1 HOUR), outtime) AS end_time,
         outtime
  FROM mimiciv_icu.icustays
  
  UNION ALL
  
  -- 递归查询，计算后续的每个小时时间段
  SELECT stay_id, DATE_ADD(start_time, INTERVAL 1 HOUR) AS start_time,
         LEAST(DATE_ADD(start_time, INTERVAL 2 HOUR), outtime) AS end_time,
         outtime
  FROM hourly_slices
  WHERE DATE_ADD(start_time, INTERVAL 1 HOUR) < outtime
)
-- 选择最终结果
SELECT stay_id, start_time, end_time
FROM hourly_slices
ORDER BY stay_id, start_time;
-- 根据时序同时对CRRT和IHD进行检查

-- 提取患者撤机7d后结局：撤机后eGFR恢复、撤机后eGFR恶化、撤机后eGFR未恢复
-- 仅关注患者的撤机和患者的肾功能预后，给出撤机时间和撤机一段时间后患者有无eGFR下降的标识
-- 撤机7d内出现eGFR下降，进行控制





