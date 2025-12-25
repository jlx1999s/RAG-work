"""医疗风险评估工具集"""

from langchain_core.tools import tool
from typing import Optional


@tool
def diabetes_risk_assessment(
    age: int,
    bmi: float,
    waist_circumference: Optional[float] = None,
    blood_pressure: str = "正常",
    family_history: bool = False,
    physical_activity: str = "适量",
    smoking: bool = False,
    diet_quality: str = "良好"
) -> dict:
    """糖尿病风险评估工具
    
    基于多个风险因素评估用户患2型糖尿病的风险等级。
    
    Args:
        age: 年龄（岁）
        bmi: 体重指数（BMI = 体重kg / 身高m²）
        waist_circumference: 腰围（厘米），可选
        blood_pressure: 血压水平，可选值：正常/偏高/高血压
        family_history: 是否有糖尿病家族史（父母、兄弟姐妹）
        physical_activity: 体育活动水平，可选值：不足/适量/充足
        smoking: 是否吸烟
        diet_quality: 饮食质量，可选值：差/一般/良好
    
    Returns:
        dict: 包含风险等级、风险评分和健康建议的评估结果
    """
    risk_score = 0
    risk_factors = []
    recommendations = []
    
    # 1. 年龄评分
    if age >= 65:
        risk_score += 30
        risk_factors.append("年龄≥65岁（高风险因素）")
    elif age >= 45:
        risk_score += 20
        risk_factors.append("年龄45-64岁（中等风险因素）")
    elif age >= 35:
        risk_score += 10
        risk_factors.append("年龄35-44岁（低风险因素）")
    
    # 2. BMI评分
    if bmi >= 28:
        risk_score += 30
        risk_factors.append(f"BMI {bmi:.1f}（肥胖，高风险）")
        recommendations.append("建议控制体重，目标BMI < 24")
    elif bmi >= 24:
        risk_score += 20
        risk_factors.append(f"BMI {bmi:.1f}（超重，中等风险）")
        recommendations.append("建议适度减重，控制饮食")
    else:
        risk_factors.append(f"BMI {bmi:.1f}（正常）")
    
    # 3. 腰围评分（腹型肥胖）
    if waist_circumference:
        # 男性 > 90cm，女性 > 85cm 为腹型肥胖（这里简化处理）
        if waist_circumference > 90:
            risk_score += 15
            risk_factors.append(f"腰围 {waist_circumference}cm（腹型肥胖）")
            recommendations.append("建议减少腹部脂肪，加强核心运动")
    
    # 4. 血压评分
    if blood_pressure == "高血压":
        risk_score += 25
        risk_factors.append("高血压（高风险因素）")
        recommendations.append("建议控制血压，定期监测")
    elif blood_pressure == "偏高":
        risk_score += 15
        risk_factors.append("血压偏高（中等风险因素）")
        recommendations.append("注意监测血压，减少盐分摄入")
    
    # 5. 家族史评分（最重要的因素之一）
    if family_history:
        risk_score += 35
        risk_factors.append("有糖尿病家族史（高风险因素）")
        recommendations.append("建议每年进行血糖筛查")
    
    # 6. 体育活动评分
    if physical_activity == "不足":
        risk_score += 20
        risk_factors.append("体育活动不足")
        recommendations.append("建议每周至少150分钟中等强度运动")
    elif physical_activity == "适量":
        risk_factors.append("体育活动适量")
    else:
        risk_score -= 5  # 充足运动可以降低风险
        risk_factors.append("体育活动充足（保护因素）")
    
    # 7. 吸烟评分
    if smoking:
        risk_score += 15
        risk_factors.append("吸烟（风险因素）")
        recommendations.append("强烈建议戒烟")
    
    # 8. 饮食质量评分
    if diet_quality == "差":
        risk_score += 15
        risk_factors.append("饮食质量差")
        recommendations.append("建议改善饮食结构，减少高糖高脂食物")
    elif diet_quality == "一般":
        risk_score += 5
        risk_factors.append("饮食质量一般")
        recommendations.append("建议优化饮食，增加蔬菜水果摄入")
    
    # 确定风险等级
    if risk_score >= 80:
        risk_level = "高风险"
        risk_color = "🔴"
        alert_message = "您的糖尿病风险较高，强烈建议尽快就医进行全面检查！"
    elif risk_score >= 50:
        risk_level = "中高风险"
        risk_color = "🟠"
        alert_message = "您的糖尿病风险偏高，建议定期体检并改善生活方式。"
    elif risk_score >= 30:
        risk_level = "中等风险"
        risk_color = "🟡"
        alert_message = "您有一定的糖尿病风险，建议保持健康生活方式并定期检查。"
    else:
        risk_level = "低风险"
        risk_color = "🟢"
        alert_message = "您目前糖尿病风险较低，请继续保持健康的生活方式。"
    
    # 通用建议
    base_recommendations = [
        "定期监测空腹血糖和糖化血红蛋白",
        "保持健康体重（BMI 18.5-23.9）",
        "均衡饮食，控制糖分和脂肪摄入",
        "每周至少150分钟中等强度有氧运动",
        "保证充足睡眠，减少压力"
    ]
    
    # 合并建议（去重）
    all_recommendations = list(set(recommendations + base_recommendations))
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_color": risk_color,
        "alert_message": alert_message,
        "risk_factors": risk_factors,
        "recommendations": all_recommendations[:6],  # 限制最多6条建议
        "next_steps": [
            "建议咨询内分泌科医生进行专业评估" if risk_score >= 50 else "建议每年进行健康体检",
            "可进行口服葡萄糖耐量试验（OGTT）" if risk_score >= 50 else "可进行空腹血糖检测",
            "建立健康档案，跟踪风险因素变化"
        ]
    }


@tool
def hypertension_risk_assessment(
    age: int,
    systolic_bp: int,
    diastolic_bp: int,
    bmi: Optional[float] = None,
    family_history: bool = False,
    smoking: bool = False,
    salt_intake: str = "适量",
    physical_activity: str = "适量",
    alcohol_consumption: str = "不饮酒"
) -> dict:
    """高血压风险评估工具
    
    基于血压值和多个风险因素评估高血压风险等级。
    
    Args:
        age: 年龄（岁）
        systolic_bp: 收缩压（mmHg）
        diastolic_bp: 舒张压（mmHg）
        bmi: 体重指数（BMI），可选
        family_history: 是否有高血压家族史
        smoking: 是否吸烟
        salt_intake: 盐分摄入量，可选值：少/适量/过多
        physical_activity: 体育活动水平，可选值：不足/适量/充足
        alcohol_consumption: 饮酒情况，可选值：不饮酒/适量/过量
    
    Returns:
        dict: 包含风险等级、风险评分和健康建议的评估结果
    """
    risk_score = 0
    risk_factors = []
    recommendations = []
    
    # 1. 血压分级（中国高血压指南标准）
    if systolic_bp >= 180 or diastolic_bp >= 110:
        bp_level = "3级高血压（重度）"
        risk_score += 50
        risk_factors.append(f"血压 {systolic_bp}/{diastolic_bp} mmHg（3级高血压，极高风险！）")
        recommendations.append("⚠️ 紧急建议：立即就医，需要药物治疗")
    elif systolic_bp >= 160 or diastolic_bp >= 100:
        bp_level = "2级高血压（中度）"
        risk_score += 40
        risk_factors.append(f"血压 {systolic_bp}/{diastolic_bp} mmHg（2级高血压，高风险）")
        recommendations.append("建议尽快就医，可能需要药物治疗")
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        bp_level = "1级高血压（轻度）"
        risk_score += 30
        risk_factors.append(f"血压 {systolic_bp}/{diastolic_bp} mmHg（1级高血压）")
        recommendations.append("建议就医评估，改善生活方式")
    elif systolic_bp >= 130 or diastolic_bp >= 85:
        bp_level = "正常高值"
        risk_score += 15
        risk_factors.append(f"血压 {systolic_bp}/{diastolic_bp} mmHg（正常高值，需警惕）")
        recommendations.append("建议密切监测血压，预防高血压")
    else:
        bp_level = "正常血压"
        risk_factors.append(f"血压 {systolic_bp}/{diastolic_bp} mmHg（正常范围）")
    
    # 2. 年龄评分
    if age >= 65:
        risk_score += 25
        risk_factors.append("年龄≥65岁（高风险因素）")
    elif age >= 55:
        risk_score += 15
        risk_factors.append("年龄55-64岁（中等风险因素）")
    elif age >= 45:
        risk_score += 10
        risk_factors.append("年龄45-54岁（低风险因素）")
    
    # 3. BMI评分
    if bmi:
        if bmi >= 28:
            risk_score += 20
            risk_factors.append(f"BMI {bmi:.1f}（肥胖，高风险）")
            recommendations.append("建议减重，目标BMI < 24")
        elif bmi >= 24:
            risk_score += 10
            risk_factors.append(f"BMI {bmi:.1f}（超重）")
            recommendations.append("建议适度减重")
        else:
            risk_factors.append(f"BMI {bmi:.1f}（正常）")
    
    # 4. 家族史评分（重要风险因素）
    if family_history:
        risk_score += 25
        risk_factors.append("有高血压家族史（高风险因素）")
        recommendations.append("建议每3-6个月测量血压")
    
    # 5. 吸烟评分
    if smoking:
        risk_score += 20
        risk_factors.append("吸烟（重要风险因素）")
        recommendations.append("强烈建议戒烟，吸烟显著增加心血管疾病风险")
    
    # 6. 盐分摄入评分
    if salt_intake == "过多":
        risk_score += 15
        risk_factors.append("盐分摄入过多")
        recommendations.append("减少盐分摄入至每天<6克（约1茶匙）")
    elif salt_intake == "适量":
        risk_factors.append("盐分摄入适量")
    else:  # 少
        risk_score -= 5
        risk_factors.append("盐分摄入较少（保护因素）")
    
    # 7. 体育活动评分
    if physical_activity == "不足":
        risk_score += 15
        risk_factors.append("体育活动不足")
        recommendations.append("建议每周至少150分钟中等强度有氧运动")
    elif physical_activity == "适量":
        risk_factors.append("体育活动适量")
    else:  # 充足
        risk_score -= 5
        risk_factors.append("体育活动充足（保护因素）")
    
    # 8. 饮酒评分
    if alcohol_consumption == "过量":
        risk_score += 15
        risk_factors.append("过量饮酒（风险因素）")
        recommendations.append("建议限制饮酒或戒酒")
    elif alcohol_consumption == "适量":
        risk_factors.append("适量饮酒")
    else:
        risk_factors.append("不饮酒")
    
    # 确定风险等级
    if risk_score >= 90:
        risk_level = "极高风险"
        risk_color = "🔴"
        alert_message = "您的高血压风险极高，强烈建议立即就医进行全面检查和治疗！"
    elif risk_score >= 60:
        risk_level = "高风险"
        risk_color = "🟠"
        alert_message = "您的高血压风险较高，建议尽快就医进行专业评估。"
    elif risk_score >= 30:
        risk_level = "中等风险"
        risk_color = "🟡"
        alert_message = "您有一定的高血压风险，建议改善生活方式并定期监测血压。"
    else:
        risk_level = "低风险"
        risk_color = "🟢"
        alert_message = "您目前高血压风险较低，请继续保持健康的生活方式。"
    
    # 通用建议
    base_recommendations = [
        "定期监测血压（每周至少1次）",
        "限盐限油，低脂饮食",
        "保持健康体重",
        "规律作息，充足睡眠",
        "学会情绪管理，减轻压力"
    ]
    
    # 合并建议（去重）
    all_recommendations = list(set(recommendations + base_recommendations))
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_color": risk_color,
        "alert_message": alert_message,
        "bp_classification": bp_level,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "risk_factors": risk_factors,
        "recommendations": all_recommendations[:7],  # 限制最多7条建议
        "next_steps": [
            "建议咨询心血管科医生进行专业评估" if risk_score >= 60 else "建议定期体检",
            "可进行24小时动态血压监测" if risk_score >= 60 else "建议在家自测血压",
            "建立血压监测档案，记录每日血压值"
        ]
    }
