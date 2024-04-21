/*
数据处理代码
查询过程中大量参考了concepts的数据，地址为：https://github.com/MIT-LCP/mimic-code/tree/v2.1.0/mimic-iv/concepts
Note: 为了方便记录和储存，在这里构建了新的schema: mimiciv_control
查询1. 患者stay_id及其过程中出现的AKI最高分级
查询2. 患者的状态指标信息（已修改。删去了可能导致大量空白字段的时间窗连表，可以分字段下载后直接进行表拼接）
查询3. 获得AKI患者的最后一个撤机时间点
*/

---- 查询1.患者stay_id及其过程中出现的AKI最高分级
-- output: 输出aki相关信息（无论cr、uo）
DROP TABLE IF EXISTS mimiciv_control.aki_level;
CREATE TABLE mimiciv_control.aki_level AS
SELECT subject_id, hadm_id, stay_id, charttime, aki_stage
FROM mimiciv_derived.kdigo_stages;


--创建aki_stage的最大值
DROP TABLE IF EXISTS mimiciv_control.max_aki_stage;
CREATE TABLE mimiciv_control.max_aki_stage AS
SELECT subject_id, hadm_id, stay_id, MAX(aki_stage) AS max_aki_stage
FROM mimiciv_control.aki_level
GROUP BY subject_id, hadm_id, stay_id
HAVING MAX(aki_stage) > 0;
/* 0409新增备注
筛选AKI最高分级大于0的患者，共42258个，与皓玮提取数量不同，需要进行检查。
来源于concepts部分，需要结合KDIGO2012标准进行检查
*/

---- 查询2：患者的状态指标信息（查询过程较慢，可以先建索引或分开字段查询）
-- sl本身没什么意义，中间过程的命名
-- 血气指标
SELECT
  sl.stay_id,
  sl.subject_id,
  bg.ph,
  bg.temperature,
  bg.lactate,
  bg.calcium,
  bg.so2,
  bg.po2,
  bg.hemoglobin,
  bg.charttime as time
FROM
  mimiciv_control.max_aki_stage sl
LEFT JOIN mimiciv_derived.bg bg ON sl.subject_id = bg.subject_id;

-- 提取 mimiciv_derived.chemistry 表的 bicarbonate, bun, creatinine, sodium, potassium 列
SELECT
  sl.stay_id,
  sl.subject_id,
  chem.bicarbonate,
  chem.bun,
  chem.creatinine,
  chem.sodium,
  chem.potassium,
  chem.charttime as time
FROM
  mimiciv_control.max_aki_stage sl
LEFT JOIN mimiciv_derived.chemistry chem ON sl.subject_id = chem.subject_id;

-- 提取 mimiciv_derived.complete_blood_count 表的 hemoglobin, wbc 列
SELECT
  sl.stay_id,
  sl.subject_id,
  cbc.hemoglobin,
  cbc.wbc,
  cbc.charttime as time
FROM
  mimiciv_control.max_aki_stage sl
LEFT JOIN mimiciv_derived.complete_blood_count cbc ON sl.subject_id = cbc.subject_id;

-- 提取 mimiciv_derived.inflammation 表的 crp 列
SELECT
  sl.stay_id,
  sl.subject_id,
  infl.crp,
  infl.charttime as time
FROM
  mimiciv_control.max_aki_stage sl
LEFT JOIN mimiciv_derived.inflammation infl ON sl.subject_id = infl.subject_id;


-- 提取 mimiciv_derived.kdigo_stages 表的 creat, uo_rt_6hr, uo_rt_12hr, uo_rt_24hr 列
SELECT
  sl.stay_id,
  sl.subject_id,
  kd.creat,
  kd.uo_rt_24hr,
  kd.charttime as time
FROM
  mimiciv_control.max_aki_stage sl
LEFT JOIN mimiciv_derived.kdigo_stages kd ON sl.stay_id = kd.stay_id;

-- 提取 mimiciv_hosp.patients 表的 gender, anchor_age 列
SELECT
  sl.subject_id,
	sl.stay_id,
  pat.gender,
  pat.anchor_age,
  idl.admission_age
FROM
  mimiciv_control.max_aki_stage sl
LEFT JOIN mimiciv_hosp.patients pat ON sl.subject_id = pat.subject_id
LEFT JOIn   mimiciv_derived.icustay_detail idl on sl.subject_id = idl.subject_id;


-- 提取 mimiciv_derived.sofa 表的 meanbp_min, sofa_24hours 列
SELECT
  sl.stay_id,
  sl.subject_id,
  sofa.meanbp_min,
  sofa.sofa_24hours,
  sofa.endtime as time
FROM
  mimiciv_control.max_aki_stage sl
LEFT JOIN mimiciv_derived.sofa sofa ON sl.stay_id = sofa.stay_id;

-- 提取 mimiciv_derived.vitalsign 表的 heart_rate, sbp, dbp, mbp 列
SELECT
  sl.stay_id,
  sl.subject_id,
  vs.heart_rate,
  vs.sbp,
  vs.dbp,
  vs.mbp,
  vs.charttime as time
FROM
  mimiciv_control.max_aki_stage sl
LEFT JOIN mimiciv_derived.vitalsign vs ON sl.stay_id = vs.stay_id;

-- 提取 mimiciv_derived.weight_durations 表的 weight 列

SELECT
  sl.stay_id,
  sl.subject_id,
  wd.weight,
 	wd.endtime as time
FROM
  mimiciv_control.max_aki_stage sl
LEFT JOIN mimiciv_derived.weight_durations wd ON sl.stay_id = wd.stay_id;




---- 查询3. 获得AKI患者的最后一个撤机时间点
-- IHD
select stay_id, max(endtime) from mimiciv_icu.procedureevents where itemid = 225441
group by stay_id;

-- CRRT
select stay_id, max(endtime) from mimiciv_icu.procedureevents where itemid = 225802
group by stay_id;

