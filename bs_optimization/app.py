"""
Flask web application for Copper Production Optimization
Provides a simple interface to optimize process parameters and compare electrolyzer economics
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from optimal_tea import OptimalTEA, calculate_npv, calculate_irr, calculate_payback_period

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page with input sliders"""
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Run optimization with user inputs and return results.

    Expects JSON with:
        - feed_grade: float (0.002 to 0.01, representing 0.2% to 1%)
        - concentrate_grade: float (0.15 to 0.35, representing 15% to 35%)
        - testing_particle_size: float (50 to 150 microns)
        - target_tonnage: int (copper production in tons/year)

    Returns JSON with:
        - Without electrolyzer: NPV, IRR, payback, parameters
        - With electrolyzer: NPV, IRR, payback, parameters
        - NPV improvement
    """
    try:
        data = request.get_json()

        # Extract inputs
        feed_grade = float(data.get('feed_grade', 0.006))  # Default 0.6%
        concentrate_grade = float(data.get('concentrate_grade', 0.30))  # Default 30%
        testing_particle_size = float(data.get('testing_particle_size', 100))  # Default 100 microns
        target_tonnage = int(data.get('target_tonnage', 10000))  # Default 10,000 tons/year
        max_iterations = int(data.get('max_iterations', 30))  # Optimization iterations

        # Validate inputs
        # Check 1: If BOTH feed grade AND concentrate grade are at LOWER extremes (poor ore quality)
        # Check 2: If all three feed parameters are at extremes (unrealistic scenario)

        feed_grade_low_extreme = feed_grade <= 0.0028  # Feed grade <= 0.3%
        concentrate_grade_low_extreme = concentrate_grade <= 0.18  # Concentrate grade <= 18%

        # If BOTH feed grade and concentrate grade are at their lower extremes, reject
        if feed_grade_low_extreme and concentrate_grade_low_extreme:
            return jsonify({
                'success': False,
                'error': f'At a feed grade of {feed_grade*100:.2f}% and {int(concentrate_grade*100)}% concentrate grade, ore processing is unlikely to be economically feasible. Please adjust your input parameters to more closely mirror real world feed scenarios.'
            }), 400

        # Check if all three feed parameters are at extremes (any extreme, not just lower)
        feed_grade_at_extreme = feed_grade <= 0.0029 or feed_grade >= 0.007  # <= 0.24% or >= 0.95%
        concentrate_grade_at_extreme = concentrate_grade <= 0.16 or concentrate_grade >= 0.33  # <= 16% or >= 33%
        testing_size_at_extreme = testing_particle_size <= 55  # <= 55 or >= 140 microns

        if feed_grade_at_extreme and concentrate_grade_at_extreme and testing_size_at_extreme:
            extreme_params = []
            if feed_grade <= 0.0024:
                extreme_params.append(f'Feed Grade of {feed_grade*100:.2f}%')
            elif feed_grade >= 0.0095:
                extreme_params.append(f'Feed Grade of {feed_grade*100:.2f}%')

            if concentrate_grade <= 0.16:
                extreme_params.append(f'Concentrate Grade of {int(concentrate_grade*100)}%')
            elif concentrate_grade >= 0.33:
                extreme_params.append(f'Concentrate Grade of {int(concentrate_grade*100)}%')

            if testing_particle_size <= 55:
                extreme_params.append(f'Feed Size of {int(testing_particle_size)} microns')
            elif testing_particle_size >= 140:
                extreme_params.append(f'Feed Size of {int(testing_particle_size)} microns')

            return jsonify({
                'success': False,
                'error': f'This is an unlikely feed scenario. {", ".join(extreme_params)}. Please adjust your input parameters to more closely mirror real world feed scenarios.'
            }), 400

        # Run optimization WITHOUT electrolyzer
        optimizer_no = OptimalTEA(
            target_cu_tons=target_tonnage,
            feed_grade=feed_grade,
            concentrate_grade=concentrate_grade,
            use_electrolyzer=False,
            feed_particle_size=testing_particle_size
        )
        results_no = optimizer_no.run_optimization(method='differential_evolution', maxiter=max_iterations, verbose=False)
        result_no = results_no['result']

        # Calculate financial metrics
        npv_no = calculate_npv(result_no['total_capex'], result_no['total_opex'], result_no['cu_produced_tons'])
        irr_no = calculate_irr(result_no['total_capex'], result_no['total_opex'], result_no['cu_produced_tons'])
        payback_no = calculate_payback_period(result_no['total_capex'], result_no['total_opex'], result_no['cu_produced_tons'])
        cost_per_ton_no = (result_no['total_capex']/25 + result_no['total_opex']) / result_no['cu_produced_tons'] if result_no['cu_produced_tons'] > 0 else 0

        # Run optimization WITH electrolyzer
        optimizer_yes = OptimalTEA(
            target_cu_tons=target_tonnage,
            feed_grade=feed_grade,
            concentrate_grade=concentrate_grade,
            use_electrolyzer=True,
            feed_particle_size=testing_particle_size
        )
        results_yes = optimizer_yes.run_optimization(method='differential_evolution', maxiter=max_iterations, verbose=False)
        result_yes = results_yes['result']

        # Calculate financial metrics
        npv_yes = calculate_npv(result_yes['total_capex'], result_yes['total_opex'], result_yes['cu_produced_tons'])
        irr_yes = calculate_irr(result_yes['total_capex'], result_yes['total_opex'], result_yes['cu_produced_tons'])
        payback_yes = calculate_payback_period(result_yes['total_capex'], result_yes['total_opex'], result_yes['cu_produced_tons'])
        cost_per_ton_yes = (result_yes['total_capex']/25 + result_yes['total_opex']) / result_yes['cu_produced_tons'] if result_yes['cu_produced_tons'] > 0 else 0

        # Calculate improvement
        npv_improvement = npv_yes - npv_no
        npv_improvement_pct = (npv_improvement / npv_no * 100) if npv_no != 0 else 0
        cost_per_ton_delta = cost_per_ton_yes - cost_per_ton_no

        # Get calculated parameters (includes num_cells for flotation)
        grinding_no = result_no['calculated_parameters']['grinding']
        flotation_no = result_no['calculated_parameters']['flotation']
        leaching_no = result_no['calculated_parameters']['leaching']
        sx_no = result_no['calculated_parameters']['sx']
        ew_no = result_no['calculated_parameters']['ew']

        grinding_yes = result_yes['calculated_parameters']['grinding']
        flotation_yes = result_yes['calculated_parameters']['flotation']
        leaching_yes = result_yes['calculated_parameters']['leaching']
        sx_yes = result_yes['calculated_parameters']['sx']
        ew_yes = result_yes['calculated_parameters']['ew']

        # Build response
        response = {
            'success': True,
            'inputs': {
                'feed_grade_pct': feed_grade * 100,
                'target_tonnage': target_tonnage,
                'concentrate_grade_pct': concentrate_grade * 100
            },
            'without_electrolyzer': {
                'npv': npv_no,
                'irr': irr_no * 100,  # Convert to percentage
                'payback_years': payback_no,
                'total_capex': result_no['total_capex'],
                'total_opex': result_no['total_opex'],
                'copper_produced': result_no['cu_produced_tons'],
                'ore_feed_required': result_no['ore_feed_tpa'],
                'cost_per_ton': cost_per_ton_no,
                'recoveries': {
                    'flotation': result_no['recoveries']['flotation'] * 100,
                    'leaching': result_no['recoveries']['leaching'] * 100,
                    'sx': result_no['recoveries']['sx'] * 100,
                    'ew': result_no['recoveries']['ew'] * 100,
                    'overall': result_no['recoveries']['overall'] * 100
                },
                'optimal_parameters': {
                    'grinding': grinding_no,
                    'flotation': flotation_no,
                    'leaching': leaching_no,
                    'sx': sx_no,
                    'ew': ew_no
                }
            },
            'with_electrolyzer': {
                'npv': npv_yes,
                'irr': irr_yes * 100,
                'payback_years': payback_yes,
                'total_capex': result_yes['total_capex'],
                'total_opex': result_yes['total_opex'],
                'copper_produced': result_yes['cu_produced_tons'],
                'ore_feed_required': result_yes['ore_feed_tpa'],
                'cost_per_ton': cost_per_ton_yes,
                'recoveries': {
                    'flotation': result_yes['recoveries']['flotation'] * 100,
                    'leaching': result_yes['recoveries']['leaching'] * 100,
                    'sx': result_yes['recoveries']['sx'] * 100,
                    'ew': result_yes['recoveries']['ew'] * 100,
                    'overall': result_yes['recoveries']['overall'] * 100
                },
                'optimal_parameters': {
                    'grinding': grinding_yes,
                    'flotation': flotation_yes,
                    'leaching': leaching_yes,
                    'sx': sx_yes,
                    'ew': ew_yes
                }
            },
            'comparison': {
                'npv_improvement': npv_improvement,
                'npv_improvement_pct': npv_improvement_pct,
                'cost_per_ton_delta': cost_per_ton_delta,
                'recommendation': 'Use Electrolyzer' if npv_improvement > 0 else 'Buy Acid'
            }
        }

        return jsonify(response)

    except Exception as e:
        # if e in '''"Error: Unexpected token 'I', ..."112,"irr":Infinity,""... is not valid JSON"''':
        #     return "This is not an economical "
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
