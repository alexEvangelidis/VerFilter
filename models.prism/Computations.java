/**
Copyright (C) Alexandros Evangelidis.

 VerFilter

 This file is part of VerFilter.

 VerFilter is free software: you can redistribute it and/or modify it
 under the terms of the GNU General Public License as published by the Free
 Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 VerFilter is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 more details.

 You should have received a copy of the GNU General Public License along with
 this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package models.prism;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Precision;

/**
 *
 * A collection of methods for various operations.
 *
 */

public class Computations {

	public static int getIntegralPartAsInt(double number, int decPlaces) {
		double number_round = Precision.round(number, decPlaces);
		if (number == 0) {
			return 0;
		} else if (number > 0) {
			int number_int = (int) Math.floor(number_round);
			return number_int;
		} else if (number < 0) {
			int number_int = (int) Math.ceil(number_round);
			return number_int;
		}
		return -1;
	}

	public static int getFractionalPartAsInt(double number, int decPlaces) {
		double number_fract = (getIntFracPartsAsDoubles(number, decPlaces));
		int number_frac = (int) Math.floor((number_fract) * Math.pow(10, decPlaces));
		return number_frac;
	}

	public static double getIntFracPartsAsDoubles(double number, int decPlaces) {
		number = Precision.round(number, decPlaces);
		if (number == 0) {
			return number;
		} else if (number < 0) {
			double integral = Math.ceil(number);
			double fractional = number - integral;
			fractional = Precision.round(fractional, decPlaces);
			return fractional;
		} else if (number > 0) {
			double integral = Math.floor(number);
			double fractional = number - integral;
			fractional = Precision.round(fractional, decPlaces);
			return fractional;
		}
		return -1;
	}

	public static double[] calcTransProbAndExpectedVal(NormalDistribution nd, int gLevel, int decPlaces) {

		double mean = nd.getMean();
		double stdNoise = nd.getStandardDeviation();
		NormalDistribution standard = new NormalDistribution(0, 1);

		if (gLevel == 2) {
			double a1_unst = Double.NEGATIVE_INFINITY;
			double b1_unst = mean;
			double a2_unst = mean;
			double b2_unst = Double.POSITIVE_INFINITY;

			// standardise a and b for left part of the distribution
			double a1_std = (a1_unst - mean) / stdNoise;
			double b1_std = (b1_unst - mean) / stdNoise;
			double a2_std = (a2_unst - mean) / stdNoise;
			double b2_std = (b2_unst - mean) / stdNoise;

			double small_phi_a1 = standard.density(a1_std);
			double small_phi_b1 = standard.density(b1_std);
			double small_phi_a2 = standard.density(a2_std);
			double small_phi_b2 = standard.density(b2_std);

			double big_phi_a1 = standard.cumulativeProbability(a1_std);
			double big_phi_b1 = standard.cumulativeProbability(b1_std);
			double big_phi_a2 = standard.cumulativeProbability(a2_std);
			double big_phi_b2 = standard.cumulativeProbability(b2_std);

			double expected_value_from_a1_to_b1 = mean
					+ stdNoise * ((small_phi_a1 - small_phi_b1) / (big_phi_b1 - big_phi_a1));
			double expected_value_from_a2_to_b2 = mean
					+ stdNoise * ((small_phi_a2 - small_phi_b2) / (big_phi_b2 - big_phi_a2));

			double trans_prob_from_a1_to_b1 = standard.cumulativeProbability(b1_std)
					- standard.cumulativeProbability(a1_std);
			double trans_prob_from_a2_to_b2 = standard.cumulativeProbability(b2_std)
					- standard.cumulativeProbability(a2_std);

			double[] transProbsExp = { trans_prob_from_a1_to_b1, expected_value_from_a1_to_b1, trans_prob_from_a2_to_b2,
					expected_value_from_a2_to_b2 };

			return transProbsExp;
		}

		else if (gLevel == 4) {
			double a1_unst = Double.NEGATIVE_INFINITY;
			double b1_unst = -2 * stdNoise;
			double a2_unst = -2 * stdNoise;
			double b2_unst = mean;
			double a3_unst = mean;
			double b3_unst = 2 * stdNoise;
			double a4_unst = 2 * stdNoise;
			double b4_unst = Double.POSITIVE_INFINITY;

			// standardise a and b for left part of the distribution
			double a1_std = (a1_unst - mean) / stdNoise;
			double b1_std = (b1_unst - mean) / stdNoise;
			double a2_std = (a2_unst - mean) / stdNoise;
			double b2_std = (b2_unst - mean) / stdNoise;
			double a3_std = (a3_unst - mean) / stdNoise;
			double b3_std = (b3_unst - mean) / stdNoise;
			double a4_std = (a4_unst - mean) / stdNoise;
			double b4_std = (b4_unst - mean) / stdNoise;

			// [-inf..-2σ]
			double small_phi_a1 = standard.density(a1_std);
			double small_phi_b1 = standard.density(b1_std);
			// -2σ..mean
			double small_phi_a2 = standard.density(a2_std);
			double small_phi_b2 = standard.density(b2_std);
			// mean..2σ
			double small_phi_a3 = standard.density(a3_std);
			double small_phi_b3 = standard.density(b3_std);
			// 2σ..+inf
			double small_phi_a4 = standard.density(a4_std);
			double small_phi_b4 = standard.density(b4_std);

			// [-inf..-2σ]
			double big_phi_a1 = standard.cumulativeProbability(a1_std);
			double big_phi_b1 = standard.cumulativeProbability(b1_std);
			// -2σ..mean
			double big_phi_a2 = standard.cumulativeProbability(a2_std);
			double big_phi_b2 = standard.cumulativeProbability(b2_std);
			// mean..2σ
			double big_phi_a3 = standard.cumulativeProbability(a3_std);
			double big_phi_b3 = standard.cumulativeProbability(b3_std);
			// 2σ..+inf
			double big_phi_a4 = standard.cumulativeProbability(a4_std);
			double big_phi_b4 = standard.cumulativeProbability(b4_std);

			double expected_value_from_a1_to_b1 = mean
					+ stdNoise * ((small_phi_a1 - small_phi_b1) / (big_phi_b1 - big_phi_a1));
			double expected_value_from_a2_to_b2 = mean
					+ stdNoise * ((small_phi_a2 - small_phi_b2) / (big_phi_b2 - big_phi_a2));

			double expected_value_from_a3_to_b3 = mean
					+ stdNoise * ((small_phi_a3 - small_phi_b3) / (big_phi_b3 - big_phi_a3));
			double expected_value_from_a4_to_b4 = mean
					+ stdNoise * ((small_phi_a4 - small_phi_b4) / (big_phi_b4 - big_phi_a4));

			double trans_prob_from_a1_to_b1 = nd.cumulativeProbability(b1_unst) - nd.cumulativeProbability(a1_unst);
			double trans_prob_from_a2_to_b2 = nd.cumulativeProbability(b2_unst) - nd.cumulativeProbability(a2_unst);
			double trans_prob_from_a3_to_b3 = nd.cumulativeProbability(b3_unst) - nd.cumulativeProbability(a3_unst);
			double trans_prob_from_a4_to_b4 = nd.cumulativeProbability(b4_unst) - nd.cumulativeProbability(a4_unst);

			double[] transProbsExp = { trans_prob_from_a1_to_b1, expected_value_from_a1_to_b1, trans_prob_from_a2_to_b2,
					expected_value_from_a2_to_b2, trans_prob_from_a3_to_b3, expected_value_from_a3_to_b3,
					trans_prob_from_a4_to_b4, expected_value_from_a4_to_b4 };

			return transProbsExp;
		}

		// mean included
		else if (gLevel == 3) {
			// 3 intervals now
			// [-inf..-2sigma] [-2sigma..2sigma]..[2sigma..infinity]
			double a1_unst = Double.NEGATIVE_INFINITY;
			double b1_unst = -2 * stdNoise;
			double a2_unst = -2 * stdNoise;
			double b2_unst = 2 * stdNoise;
			double a3_unst = 2 * stdNoise;
			;
			double b3_unst = Double.POSITIVE_INFINITY;

			// standardise a and b for left part of the distribution
			double a1_std = (a1_unst - mean) / stdNoise;
			double b1_std = (b1_unst - mean) / stdNoise;
			double a2_std = (a2_unst - mean) / stdNoise;
			double b2_std = (b2_unst - mean) / stdNoise;
			double a3_std = (a3_unst - mean) / stdNoise;
			double b3_std = (b3_unst - mean) / stdNoise;

			// [-inf..-2σ]
			double small_phi_a1 = standard.density(a1_std);
			double small_phi_b1 = standard.density(b1_std);
			// -2σ..2σ
			double small_phi_a2 = standard.density(a2_std);
			double small_phi_b2 = standard.density(b2_std);

			// 2σ..+inf
			double small_phi_a3 = standard.density(a3_std);
			double small_phi_b3 = standard.density(b3_std);

			// [-inf..-2σ]
			double big_phi_a1 = standard.cumulativeProbability(a1_std);
			double big_phi_b1 = standard.cumulativeProbability(b1_std);
			// -2σ..2σ
			double big_phi_a2 = standard.cumulativeProbability(a2_std);
			double big_phi_b2 = standard.cumulativeProbability(b2_std);
			// 2σ..+inf
			double big_phi_a3 = standard.cumulativeProbability(a3_std);
			double big_phi_b3 = standard.cumulativeProbability(b3_std);

			double expected_value_from_a1_to_b1 = mean
					+ stdNoise * ((small_phi_a1 - small_phi_b1) / (big_phi_b1 - big_phi_a1));
			double expected_value_from_a2_to_b2 = mean
					+ stdNoise * ((small_phi_a2 - small_phi_b2) / (big_phi_b2 - big_phi_a2));
			double expected_value_from_a3_to_b3 = mean
					+ stdNoise * ((small_phi_a3 - small_phi_b3) / (big_phi_b3 - big_phi_a3));

			double trans_prob_from_a1_to_b1 = nd.cumulativeProbability(b1_unst) - nd.cumulativeProbability(a1_unst);
			double trans_prob_from_a2_to_b2 = nd.cumulativeProbability(b2_unst) - nd.cumulativeProbability(a2_unst);
			double trans_prob_from_a3_to_b3 = nd.cumulativeProbability(b3_unst) - nd.cumulativeProbability(a3_unst);

			double[] transProbsExp = { trans_prob_from_a1_to_b1, expected_value_from_a1_to_b1, trans_prob_from_a2_to_b2,
					expected_value_from_a2_to_b2, trans_prob_from_a3_to_b3, expected_value_from_a3_to_b3 };

			return transProbsExp;

		} else if (gLevel == 4) {
			// 6 intervals
			// [-inf..-2sigma]
			// [-2sigma..-sigma]..[-sigma..m]..[m..sigma]..[sigma..2sigma]..[2sigma..+infinity]
			// [-inf..-2igma]
			double a1_unst = Double.NEGATIVE_INFINITY;
			double b1_unst = -2 * stdNoise;
			// [-2sigma..sigma]
			double a2_unst = -2 * stdNoise;
			double b2_unst = -stdNoise;
			// [-sigma..mean]
			double a3_unst = -stdNoise;
			double b3_unst = mean;
			// [mean..sigma]
			double a4_unst = mean;
			double b4_unst = stdNoise;
			// [sigma..2sigma]
			double a5_unst = stdNoise;
			double b5_unst = 2 * stdNoise;
			// [2sigma..inf]
			double a6_unst = 2 * stdNoise;
			double b6_unst = Double.POSITIVE_INFINITY;

			// standardise a1 and b1
			double a1_std = (a1_unst - mean) / stdNoise;
			double b1_std = (b1_unst - mean) / stdNoise;

			// standardise a2 and b2
			double a2_std = (a2_unst - mean) / stdNoise;
			double b2_std = (b2_unst - mean) / stdNoise;
			// standardise a3 and b3
			double a3_std = (a3_unst - mean) / stdNoise;
			double b3_std = (b3_unst - mean) / stdNoise;
			// standardise a4 and b4
			double a4_std = (a4_unst - mean) / stdNoise;
			double b4_std = (b4_unst - mean) / stdNoise;
			// standardise a5 and b5
			double a5_std = (a5_unst - mean) / stdNoise;
			double b5_std = (b5_unst - mean) / stdNoise;
			// standardise a6 and b6
			double a6_std = (a6_unst - mean) / stdNoise;
			double b6_std = (b6_unst - mean) / stdNoise;

			// [-inf..-2sigma]
			double small_phi_a1 = standard.density(a1_std);
			double small_phi_b1 = standard.density(b1_std);

			// [-2sigma..-sigma]
			double small_phi_a2 = standard.density(a2_std);
			double small_phi_b2 = standard.density(b2_std);
			// [-sigma..mean]
			double small_phi_a3 = standard.density(a3_std);
			double small_phi_b3 = standard.density(b3_std);
			// [mean..sigma]
			double small_phi_a4 = standard.density(a4_std);
			double small_phi_b4 = standard.density(b4_std);
			// [sigma..2sigma]
			double small_phi_a5 = standard.density(a5_std);
			double small_phi_b5 = standard.density(b5_std);
			// [2sigma..+inf]
			double small_phi_a6 = standard.density(a6_std);
			double small_phi_b6 = standard.density(b6_std);

			// [-inf..-2sigma]
			double big_phi_a1 = standard.cumulativeProbability(a1_std);
			double big_phi_b1 = standard.cumulativeProbability(b1_std);
			// [-2sigma..sigma]
			double big_phi_a2 = standard.cumulativeProbability(a2_std);
			double big_phi_b2 = standard.cumulativeProbability(b2_std);
			// [-sigma..mean]
			double big_phi_a3 = standard.cumulativeProbability(a3_std);
			double big_phi_b3 = standard.cumulativeProbability(b3_std);
			// [mean..sigma]
			double big_phi_a4 = standard.cumulativeProbability(a4_std);
			double big_phi_b4 = standard.cumulativeProbability(b4_std);
			// [sigma..2sigma]
			double big_phi_a5 = standard.cumulativeProbability(a5_std);
			double big_phi_b5 = standard.cumulativeProbability(b5_std);
			// [2sigma..+inf]
			double big_phi_a6 = standard.cumulativeProbability(a6_std);
			double big_phi_b6 = standard.cumulativeProbability(b6_std);

			double expected_value_from_a1_to_b1 = mean
					+ stdNoise * ((small_phi_a1 - small_phi_b1) / (big_phi_b1 - big_phi_a1));
			double expected_value_from_a2_to_b2 = mean
					+ stdNoise * ((small_phi_a2 - small_phi_b2) / (big_phi_b2 - big_phi_a2));
			double expected_value_from_a3_to_b3 = mean
					+ stdNoise * ((small_phi_a3 - small_phi_b3) / (big_phi_b3 - big_phi_a3));
			double expected_value_from_a4_to_b4 = mean
					+ stdNoise * ((small_phi_a4 - small_phi_b4) / (big_phi_b4 - big_phi_a4));
			double expected_value_from_a5_to_b5 = mean
					+ stdNoise * ((small_phi_a5 - small_phi_b5) / (big_phi_b5 - big_phi_a5));
			double expected_value_from_a6_to_b6 = mean
					+ stdNoise * ((small_phi_a6 - small_phi_b6) / (big_phi_b6 - big_phi_a6));

			double trans_prob_from_a1_to_b1 = nd.cumulativeProbability(b1_unst) - nd.cumulativeProbability(a1_unst);
			double trans_prob_from_a2_to_b2 = nd.cumulativeProbability(b2_unst) - nd.cumulativeProbability(a2_unst);
			double trans_prob_from_a3_to_b3 = nd.cumulativeProbability(b3_unst) - nd.cumulativeProbability(a3_unst);
			double trans_prob_from_a4_to_b4 = nd.cumulativeProbability(b4_unst) - nd.cumulativeProbability(a4_unst);
			double trans_prob_from_a5_to_b5 = nd.cumulativeProbability(b5_unst) - nd.cumulativeProbability(a5_unst);
			double trans_prob_from_a6_to_b6 = nd.cumulativeProbability(b6_unst) - nd.cumulativeProbability(a6_unst);

			double[] transProbsExp = { trans_prob_from_a1_to_b1, expected_value_from_a1_to_b1, trans_prob_from_a2_to_b2,
					expected_value_from_a2_to_b2, trans_prob_from_a3_to_b3, expected_value_from_a3_to_b3,
					trans_prob_from_a4_to_b4, expected_value_from_a4_to_b4, trans_prob_from_a5_to_b5,
					expected_value_from_a5_to_b5, trans_prob_from_a6_to_b6, expected_value_from_a6_to_b6 };

			return transProbsExp;
		} else if (gLevel == 5) {
			// 5 intervals

			// // [-inf..-2sigma]
			// [-2sigma..-sigma]..[-sigma..sigma]..[sigma..2sigma]..[2sigma..+infinity]
			// [-inf..-2igma]
			double a1_unst = Double.NEGATIVE_INFINITY;
			double b1_unst = -2 * stdNoise;
			// [-2sigma..-sigma]
			double a2_unst = -2 * stdNoise;
			double b2_unst = -stdNoise;
			// [-sigma..sigma]
			double a3_unst = -stdNoise;
			double b3_unst = stdNoise;
			// [sigma..+2sigma]
			double a4_unst = stdNoise;
			double b4_unst = 2 * stdNoise;

			// [+2sigma..+inf]
			double a5_unst = 2 * stdNoise;

			double b5_unst = Double.POSITIVE_INFINITY;

			// standardise a1 and b1
			double a1_std = (a1_unst - mean) / stdNoise;
			double b1_std = (b1_unst - mean) / stdNoise;
			// standardise a2 and b2
			double a2_std = (a2_unst - mean) / stdNoise;
			double b2_std = (b2_unst - mean) / stdNoise;
			// standardise a3 and b3
			double a3_std = (a3_unst - mean) / stdNoise;
			double b3_std = (b3_unst - mean) / stdNoise;
			// standardise a4 and b4
			double a4_std = (a4_unst - mean) / stdNoise;
			double b4_std = (b4_unst - mean) / stdNoise;
			// standardise a5 and b5
			double a5_std = (a5_unst - mean) / stdNoise;
			double b5_std = (b5_unst - mean) / stdNoise;

			// [-inf..-2sigma]
			double small_phi_a1 = standard.density(a1_std);
			double small_phi_b1 = standard.density(b1_std);
			// [-2sigma..-sigma]
			double small_phi_a2 = standard.density(a2_std);
			double small_phi_b2 = standard.density(b2_std);
			// [-sigma..+sigma]
			double small_phi_a3 = standard.density(a3_std);
			double small_phi_b3 = standard.density(b3_std);
			// [+sigma..+2sigma]
			double small_phi_a4 = standard.density(a4_std);
			double small_phi_b4 = standard.density(b4_std);
			// [+2sigma..+inf]
			double small_phi_a5 = standard.density(a5_std);
			double small_phi_b5 = standard.density(b5_std);

			// [-inf..-2sigma]
			double big_phi_a1 = standard.cumulativeProbability(a1_std);
			double big_phi_b1 = standard.cumulativeProbability(b1_std);
			// [-2sigma..-sigma]
			double big_phi_a2 = standard.cumulativeProbability(a2_std);
			double big_phi_b2 = standard.cumulativeProbability(b2_std);
			// [-sigma..+sigma]
			double big_phi_a3 = standard.cumulativeProbability(a3_std);
			double big_phi_b3 = standard.cumulativeProbability(b3_std);
			// [+sigma..+2sigma]
			double big_phi_a4 = standard.cumulativeProbability(a4_std);
			double big_phi_b4 = standard.cumulativeProbability(b4_std);
			// [2sigma..+inf]
			double big_phi_a5 = standard.cumulativeProbability(a5_std);
			double big_phi_b5 = standard.cumulativeProbability(b5_std);

			double expected_value_from_a1_to_b1 = mean
					+ stdNoise * ((small_phi_a1 - small_phi_b1) / (big_phi_b1 - big_phi_a1));
			double expected_value_from_a2_to_b2 = mean
					+ stdNoise * ((small_phi_a2 - small_phi_b2) / (big_phi_b2 - big_phi_a2));
			double expected_value_from_a3_to_b3 = mean
					+ stdNoise * ((small_phi_a3 - small_phi_b3) / (big_phi_b3 - big_phi_a3));
			double expected_value_from_a4_to_b4 = mean
					+ stdNoise * ((small_phi_a4 - small_phi_b4) / (big_phi_b4 - big_phi_a4));
			double expected_value_from_a5_to_b5 = mean
					+ stdNoise * ((small_phi_a5 - small_phi_b5) / (big_phi_b5 - big_phi_a5));

			double trans_prob_from_a1_to_b1 = nd.cumulativeProbability(b1_unst) - nd.cumulativeProbability(a1_unst);
			double trans_prob_from_a2_to_b2 = nd.cumulativeProbability(b2_unst) - nd.cumulativeProbability(a2_unst);
			double trans_prob_from_a3_to_b3 = nd.cumulativeProbability(b3_unst) - nd.cumulativeProbability(a3_unst);
			double trans_prob_from_a4_to_b4 = nd.cumulativeProbability(b4_unst) - nd.cumulativeProbability(a4_unst);
			double trans_prob_from_a5_to_b5 = nd.cumulativeProbability(b5_unst) - nd.cumulativeProbability(a5_unst);

			double[] transProbsExp = { trans_prob_from_a1_to_b1, expected_value_from_a1_to_b1, trans_prob_from_a2_to_b2,
					expected_value_from_a2_to_b2, trans_prob_from_a3_to_b3, expected_value_from_a3_to_b3,
					trans_prob_from_a4_to_b4, expected_value_from_a4_to_b4, trans_prob_from_a5_to_b5,
					expected_value_from_a5_to_b5 };

			return transProbsExp;

		} else if (gLevel == 6) {
			// 6 intervals
			// [-inf..-2sigma]
			// [-2sigma..-sigma]..[-sigma..m]..[m..sigma]..[sigma..2sigma]..[2sigma..+infinity]

			// [-inf..-2igma]
			double a1_unst = Double.NEGATIVE_INFINITY;
			double b1_unst = -(2 * stdNoise);
			// [-2sigma..-sigma]
			double a2_unst = -(2 * stdNoise);
			double b2_unst = -stdNoise;
			// [-sigma..+sigma]
			double a3_unst = -stdNoise;
			double b3_unst = stdNoise;
			// [sigma..+2sigma]
			double a4_unst = stdNoise;
			double b4_unst = (2 * stdNoise);
			// [2sigma..inf]
			double a5_unst = (2d * stdNoise);
			double b5_unst = Double.POSITIVE_INFINITY;

			// standardise a1 and b1
			double a1_std = (a1_unst - mean) / stdNoise;
			double b1_std = (b1_unst - mean) / stdNoise;
			// standardise a2 and b2
			double a2_std = (a2_unst - mean) / stdNoise;
			double b2_std = (b2_unst - mean) / stdNoise;
			// standardise a3 and b3
			double a3_std = (a3_unst - mean) / stdNoise;
			double b3_std = (b3_unst - mean) / stdNoise;
			// standardise a4 and b4
			double a4_std = (a4_unst - mean) / stdNoise;
			double b4_std = (b4_unst - mean) / stdNoise;
			// standardise a5 and b5
			double a5_std = (a5_unst - mean) / stdNoise;
			double b5_std = (b5_unst - mean) / stdNoise;

			// [-inf..-2sigma]
			double small_phi_a1 = standard.density(a1_std);
			double small_phi_b1 = standard.density(b1_std);
			// [-2sigma..-sigma]
			double small_phi_a2 = standard.density(a2_std);
			double small_phi_b2 = standard.density(b2_std);
			// [-sigma..+sigma]
			double small_phi_a3 = standard.density(a3_std);
			double small_phi_b3 = standard.density(b3_std);
			// [+sigma..+2sigma]
			double small_phi_a4 = standard.density(a4_std);
			double small_phi_b4 = standard.density(b4_std);
			// [+2sigma..+inf]
			double small_phi_a5 = standard.density(a5_std);
			double small_phi_b5 = standard.density(b5_std);

			// [-inf..-2sigma]
			double big_phi_a1 = standard.cumulativeProbability(a1_std);
			double big_phi_b1 = standard.cumulativeProbability(b1_std);
			// [-2sigma..-sigma]
			double big_phi_a2 = standard.cumulativeProbability(a2_std);
			double big_phi_b2 = standard.cumulativeProbability(b2_std);
			// [-sigma..+sigma]
			double big_phi_a3 = standard.cumulativeProbability(a3_std);
			double big_phi_b3 = standard.cumulativeProbability(b3_std);
			// [+sigma..+2sigma]
			double big_phi_a4 = standard.cumulativeProbability(a4_std);
			double big_phi_b4 = standard.cumulativeProbability(b4_std);
			// [+2sigma..+inf]
			double big_phi_a5 = standard.cumulativeProbability(a5_std);
			double big_phi_b5 = standard.cumulativeProbability(b5_std);

			double expected_value_from_a1_to_b1 = mean
					+ stdNoise * ((small_phi_a1 - small_phi_b1) / (big_phi_b1 - big_phi_a1));
			double expected_value_from_a2_to_b2 = mean
					+ stdNoise * ((small_phi_a2 - small_phi_b2) / (big_phi_b2 - big_phi_a2));
			double expected_value_from_a3_to_b3 = mean
					+ stdNoise * ((small_phi_a3 - small_phi_b3) / (big_phi_b3 - big_phi_a3));
			double expected_value_from_a4_to_b4 = mean
					+ stdNoise * ((small_phi_a4 - small_phi_b4) / (big_phi_b4 - big_phi_a4));
			double expected_value_from_a5_to_b5 = mean
					+ stdNoise * ((small_phi_a5 - small_phi_b5) / (big_phi_b5 - big_phi_a5));

			double trans_prob_from_a1_to_b1 = nd.cumulativeProbability(b1_unst) - nd.cumulativeProbability(a1_unst);
			double trans_prob_from_a2_to_b2 = nd.cumulativeProbability(b2_unst) - nd.cumulativeProbability(a2_unst);
			double trans_prob_from_a3_to_b3 = nd.cumulativeProbability(b3_unst) - nd.cumulativeProbability(a3_unst);
			double trans_prob_from_a4_to_b4 = nd.cumulativeProbability(b4_unst) - nd.cumulativeProbability(a4_unst);
			double trans_prob_from_a5_to_b5 = nd.cumulativeProbability(b5_unst) - nd.cumulativeProbability(a5_unst);

			double[] transProbsExp = { Precision.round(trans_prob_from_a1_to_b1, 2), (expected_value_from_a1_to_b1),
					Precision.round(trans_prob_from_a2_to_b2, 2), (expected_value_from_a2_to_b2),
					Precision.round(trans_prob_from_a3_to_b3, 2), (expected_value_from_a3_to_b3),
					Precision.round(trans_prob_from_a4_to_b4, 2), (expected_value_from_a4_to_b4),
					Precision.round(trans_prob_from_a5_to_b5, 2), (expected_value_from_a5_to_b5) };

			return transProbsExp;

		}
		return null;
	}

	/**
	 * Returns upper triangular Cholesky factor C such that P = CC^T. Algorithm
	 * based on the following books: 1) Grewal, M.S., Andrews, A.P.: Kalman
	 * Filtering: Theory and Practice with MATLAB. Wiley-IEEE Press, 4th edn.
	 * (2014) 2) Maybeck, P.S.: Stochastic models, estimation, and control: Volume
	 * 1. Mathematics in science and engineering, Elsevier Science, Burlington, MA
	 * (1982)
	 *
	 * @param P, input matrix
	 * @return C, such that P=CC^T
	 */
	public static RealMatrix getUTCholeskyFactor(RealMatrix P) {

		if (P.getRowDimension() != P.getColumnDimension())
			throw new DimensionMismatchException(0, 0);
		RealMatrix C = new Array2DRowRealMatrix(P.getRowDimension(), P.getColumnDimension());
		int m = P.getColumnDimension();
		for (int j = m - 1; j >= 0; j--) {
			for (int i = j; i >= 0; i--) {
				double sigma = P.getEntry(i, j);

				for (int k = j + 1; k < m; k++) {
					sigma = sigma - C.getEntry(i, k) * C.getEntry(j, k);
				}
				C.setEntry(j, i, 0);
				if (i == j)
					C.setEntry(i, j, Math.sqrt(Math.max(0, sigma)));
				else if (C.getEntry(j, j) == 0) {
					C.setEntry(i, j, 0);
				} else
					C.setEntry(i, j, sigma / C.getEntry(j, j));
			}
		}
		return C;
	}
}
