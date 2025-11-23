"""General financial analysis tools"""

def analyze_income_expenditure(income: float, expenses: float) -> dict:
    """Comprehensive income vs expenditure analysis"""
    savings = income - expenses
    savings_rate = (savings / income * 100) if income > 0 else 0
    
    # Financial health score (0-100)
    if savings_rate >= 30:
        health_score = 90
        health_status = "Excellent"
    elif savings_rate >= 20:
        health_score = 75
        health_status = "Good"
    elif savings_rate >= 10:
        health_score = 50
        health_status = "Average"
    else:
        health_score = 25
        health_status = "Poor"
    
    return {
        "income": income,
        "expenses": expenses,
        "savings": savings,
        "savings_rate": round(savings_rate, 2),
        "financial_health_score": health_score,
        "financial_health_status": health_status,
        "recommendations": [
            "Build emergency fund (3-6 months expenses)",
            "Invest in diversified portfolio",
            "Consider insurance coverage"
        ]
    }

def investment_recommendation(age: int, risk_tolerance: str, amount: float) -> dict:
    """Recommend investment strategy based on profile"""
    risk_tolerance = risk_tolerance.lower()
    
    if risk_tolerance == "high" and age < 35:
        allocation = {
            "equity": 70,
            "debt": 20,
            "gold": 10
        }
        expected_return = "12-15%"
    elif risk_tolerance == "medium" or (age >= 35 and age < 50):
        allocation = {
            "equity": 50,
            "debt": 35,
            "gold": 15
        }
        expected_return = "10-12%"
    else:  # low risk or age >= 50
        allocation = {
            "equity": 30,
            "debt": 50,
            "gold": 20
        }
        expected_return = "8-10%"
    
    return {
        "age": age,
        "risk_tolerance": risk_tolerance,
        "investment_amount": amount,
        "recommended_allocation": allocation,
        "expected_annual_return": expected_return,
        "advice": f"Diversified portfolio suitable for {risk_tolerance} risk profile"
    }

def emergency_fund_calculator(monthly_expenses: float, months: int = 6) -> dict:
    """Calculate emergency fund requirement"""
    emergency_fund = monthly_expenses * months
    
    return {
        "monthly_expenses": monthly_expenses,
        "recommended_months": months,
        "emergency_fund_target": round(emergency_fund, 2),
        "advice": f"Keep ₹{emergency_fund:,.0f} in liquid savings for emergencies"
    }