"""Behavior checks for fitting, full-frequency coverage, anchors, and decisions."""
import json
import math
import unittest

import numpy as np
from scipy.stats import f

from reference import (CalibrationData, ErrorBudgets, InputError, align_recording,
                       fit_calibration, frequency_certificate, reflection_statistic,
                       run_test, trace_radius)


def calibration():
    design = np.tile(np.eye(2), (8, 1))
    residual = np.tile([0.01, -0.02], 8)*np.repeat([1, -1], 8)
    response = design@np.array([1.0, 0.2])+residual
    return CalibrationData(design, response, [0, 1])


def exact_fit(coefficients, positions):
    return dict(coefficients=coefficients, positions=positions,
                coefficient_error_l2=0.0, coefficient_error_l1=0.0)


class ReferenceTests(unittest.TestCase):
    def test_exact_regression_pivot_radius(self):
        data = calibration()
        result = fit_calibration(data, 0.005)
        np.testing.assert_allclose(result['coefficients'], [1.0, 0.2], atol=1e-14)
        expected_variance = np.dot(data.response-data.design@np.array([1., .2]),
                                   data.response-data.design@np.array([1., .2]))/14
        expected_ellipse = 2*expected_variance*f.ppf(.995, 2, 14)
        self.assertAlmostEqual(result['residual_variance'], expected_variance, places=17)
        self.assertAlmostEqual(result['ellipse_radius_squared'], expected_ellipse, places=15)
        self.assertAlmostEqual(result['coefficient_error_l2'], math.sqrt(expected_ellipse/8), places=15)

    def test_full_frequency_signed_position_coverage(self):
        positions = np.array([-2, 0, 3])
        first = np.array([.35, 1., -.1])
        second = np.array([.2, .8, .3])
        perturbations = [np.array([.004, -.003, .002]), np.array([-.002, .002, .001])]
        fits = []
        for coefficients, perturbation in zip([first, second], perturbations):
            error = float(np.linalg.norm(perturbation))
            fits.append(dict(coefficients=(coefficients+perturbation).tolist(), positions=positions.tolist(),
                             coefficient_error_l2=error, coefficient_error_l1=math.sqrt(3)*error))
        certificate = frequency_certificate(fits, anchors=(-1, 2), grid_points=256)
        frequencies = 2*math.pi*(np.arange(32003)+.37)/32003
        responses = [np.exp(-1j*np.outer(frequencies, positions-anchor))@coefficients
                     for coefficients,anchor in zip([first,second], [-1,2])]
        product = responses[0]*np.conjugate(responses[1])
        target = np.abs(np.sin(frequencies))*np.abs(np.imag(product))/np.abs(product)
        self.assertLessEqual(float(target.max()), certificate['phase_upper']+1e-13)
        for coefficients, response, bound in zip([first,second], responses, certificate['shape_upper']):
            true_shape = float(np.max(np.abs(response)**2)/np.dot(coefficients, coefficients))
            self.assertLessEqual(true_shape, bound+1e-13)

    def test_transfer_zero_and_norm_fallback(self):
        fits = [exact_fit([1., 1.], [0, 2])]*2
        certificate = frequency_certificate(fits, grid_points=64)
        self.assertEqual(certificate['phase_upper'], 1.0)
        self.assertGreater(certificate['universal_product_cells'], 0)
        zeros = frequency_certificate([exact_fit([0., 0.], [0, 1])]*2, grid_points=64)
        self.assertEqual(zeros['shape_upper'], [2., 2.])
        self.assertTrue(all(c['norm_floor_fallback'] for c in zeros['channels']))

    def test_anchor_alignment_and_retained_length(self):
        raw = np.column_stack([np.arange(12.), 100+np.arange(12.)])
        aligned, details = align_recording(raw, (-2, 3))
        np.testing.assert_array_equal(aligned[:, 0], np.arange(7.))
        np.testing.assert_array_equal(aligned[:, 1], 105+np.arange(7.))
        self.assertEqual(details['retained_record_length'], 7)
        self.assertEqual(details['base_time_start'], 2)
        result = run_test([calibration()]*2, np.tile(raw, (200,1)), 1., 1., [1.,1.], anchors=(-2,3), grid_points=64)
        self.assertEqual(result['alignment']['retained_record_length'], 2395)
        self.assertEqual(result['window_count'], 2394)
        expected = 1-result['rho_upper']/2395-2*math.sqrt(result['rho_upper']*math.log(200)/2395)
        self.assertAlmostEqual(result['scale_denominator'], expected)
        self.assertEqual(result['certificate']['channels'][0]['shifted_positions'], [2,3])
        self.assertEqual(result['certificate']['channels'][1]['shifted_positions'], [-3,-2])
        self.assertAlmostEqual(result['sampling_radius']['operator_norm_upper'], 2/2394)

    def test_statistic_matches_explicit_quadratic_matrix(self):
        values = np.array([[.2, -.4], [1.1,.3], [-.7,1.2], [.5,-.6], [1.2,.8]])
        length = len(values)
        n = length-1
        plus = np.zeros((n,length)); minus = np.zeros_like(plus)
        for j in range(n):
            plus[j,j:j+2] = [1,1]
            minus[j,j:j+2] = [1,-1]
        center = np.eye(n)-np.ones((n,n))/n
        cross = plus.T@center@minus/(2*n)
        matrix = np.block([[np.zeros((length,length)), cross], [cross.T,np.zeros((length,length))]])
        stacked = values.T.ravel()
        self.assertAlmostEqual(reflection_statistic(values), float(stacked@matrix@stacked), places=14)
        self.assertAlmostEqual(reflection_statistic(values), reflection_statistic(values+[2.,-3.]), places=14)
        radius = trace_radius(20., 1.3, length, 4.)
        self.assertAlmostEqual(np.linalg.norm(matrix), radius['F_n'], places=14)
        self.assertLessEqual(np.linalg.norm(matrix,2), radius['operator_norm_upper']+1e-14)

    def test_trace_minimum_and_threshold_order(self):
        previous = None
        for rho in [2., 8., 16., 64.]:
            result = trace_radius(rho, 1.2, 10000, math.log(2/.03))
            self.assertLessEqual(result['radius'], result['old_radius'])
            if previous is not None:
                self.assertGreaterEqual(result['radius'], previous)
            previous = result['radius']
        self.assertEqual(trace_radius(64., 1., 10000, 4.)['branch'], 'trace')
        self.assertEqual(trace_radius(2., 1., 10000, 4.)['branch'], 'Frobenius')

    def test_colored_error_ratio_and_abstention(self):
        record = np.column_stack([np.sin(np.arange(32.)), np.cos(np.arange(32.))])
        result = run_test([calibration()]*2, record, 1., 1., [1000.,1.], grid_points=64)
        self.assertEqual(result['rho_upper'], 1000.)
        self.assertEqual(result['status'], 'ABSTAIN')
        self.assertIsNone(result['decision'])
        self.assertIsNone(result['threshold'])
        self.assertIsNone(result['common_variance_ceiling'])

    def test_invalid_inputs_fail_closed(self):
        good = calibration()
        record = np.column_stack([np.arange(1000.), np.arange(1000.)])
        invalid_calibrations = [
            CalibrationData(np.ones((16,2)), np.ones(16), [0,1]),
            CalibrationData(good.design, good.response, [0,0]),
            CalibrationData(good.design, good.response, [0.,1.]),
            CalibrationData(good.design, good.response, [0,1], include_intercept=True),
            CalibrationData(good.design, np.full(16,np.nan), [0,1]),
            CalibrationData(np.eye(2), np.ones(2), [0,1]),
        ]
        for data in invalid_calibrations:
            with self.subTest(data=data):
                with self.assertRaises(InputError):
                    run_test([data,good], record, 1.,1.,[1.,1.])
        for kwargs in [dict(source_floor=0.), dict(source_ceiling=.5), dict(error_peak_ratios=[.9,1.]),
                       dict(anchors=(0,.5)), dict(anchors=(0,999)), dict(reflection_spacing=2),
                       dict(grid_points=0), dict(budgets=ErrorBudgets(tail=.04)),
                       dict(budgets=ErrorBudgets(variance=float('nan'))), dict(budgets=ErrorBudgets(variance='0.01')),
                       dict(budgets=ErrorBudgets(calibration=None)), dict(error_peak_ratios=None),
                       dict(source_floor='1'), dict(anchors=None), dict(recording=np.full((1000,2), np.inf))]:
            arguments = dict(calibrations=[good,good], recording=record, source_floor=1., source_ceiling=1., error_peak_ratios=[1.,1.])
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(InputError):
                    run_test(**arguments)

    def test_finite_overflow_cannot_be_masked_by_clipping(self):
        fits = [exact_fit([1e155], [0])]*2
        with np.errstate(over='ignore', invalid='ignore'):
            with self.assertRaises(InputError):
                frequency_certificate(fits, grid_points=16)
        with self.assertRaises(InputError):
            trace_radius(1e308, 0., 1000, 4.)
        design = np.tile(np.eye(2), (64,1))*1e-200
        response = np.repeat(np.where(np.arange(64)%2, 1., -1.), 2)*.001
        with self.assertRaises(InputError):
            fit_calibration(CalibrationData(design, response, [0,1]), .005)

    def test_budget_allocation_changes_the_fitted_radius(self):
        data = calibration()
        smaller_failure = fit_calibration(data, .001)
        larger_failure = fit_calibration(data, .01)
        self.assertGreater(smaller_failure['coefficient_error_l2'], larger_failure['coefficient_error_l2'])
        budget = ErrorBudgets(calibration=(.002,.003), variance=.015, tail=.03).validate()
        self.assertAlmostEqual(budget['total'], .05)
        self.assertAlmostEqual(budget['calibration_total'], .005)

    def test_no_record_split_and_input_unchanged(self):
        record = np.column_stack([np.sin(np.arange(10000.)/3), np.cos(np.arange(10000.)/3)])
        original = record.copy()
        result = run_test([calibration()]*2, record, 1.,1.,[1.,1.], grid_points=64)
        np.testing.assert_array_equal(record, original)
        self.assertEqual(result['alignment']['retained_record_length'], len(record))
        self.assertEqual(result['window_count'], len(record)-1)
        self.assertAlmostEqual(result['threshold'], result['null_tolerance']+result['sampling_radius']['radius'])
        self.assertEqual(result['rejection'], abs(result['statistic']) > result['threshold'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
