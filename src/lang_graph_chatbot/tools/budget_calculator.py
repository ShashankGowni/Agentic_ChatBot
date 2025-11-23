"""Budget calculation and analysis tools"""

def calculate_savings(income: float, expenses: float) -> dict:
    """Calculate monthly savings"""
    savings = income - expenses
    savings_percentage = (savings / income * 100) if income > 0 else 0
    
    return {
        "income": income,
        "expenses": expenses,
        "savings": savings,
        "savings_percentage": round(savings_percentage, 2)
    }

def analyze_budget(income: float, expenses_dict: dict) -> dict:
    """Analyze detailed budget breakdown"""
    total_expenses = sum(expenses_dict.values())
    savings = income - total_expenses
    savings_percentage = (savings / income * 100) if income > 0 else 0
    
    # Calculate emergency fund (3-6 months of expenses)
    emergency_fund = total_expenses * 3
    
    # Recommendations
    if savings_percentage >= 30:
        recommendation = "Excellent! You're saving well."
    elif savings_percentage >= 20:
        recommendation = "Good savings rate. Consider investing more."
    elif savings_percentage >= 10:
        recommendation = "Fair. Try to increase savings by reducing expenses."
    else:
        recommendation = "Low savings. Review and cut unnecessary expenses."
    
    return {
        "income": income,
        "total_expenses": total_expenses,
        "expense_breakdown": expenses_dict,
        "savings": savings,
        "savings_percentage": round(savings_percentage, 2),
        "emergency_fund_target": emergency_fund,
        "recommendation": recommendation
    }

def suggest_budget_plan(income: float, savings_goal: float) -> dict:
    """Suggest budget allocation using 50-30-20 rule"""
    needs = income * 0.50  # 50% for needs
    wants = income * 0.30  # 30% for wants
    savings = income * 0.20  # 20% for savings
    
    return {
        "income": income,
        "suggested_needs": needs,
        "suggested_wants": wants,
        "suggested_savings": savings,
        "custom_savings_goal": savings_goal,
        "rule": "50-30-20 Rule (Needs-Wants-Savings)"
    }