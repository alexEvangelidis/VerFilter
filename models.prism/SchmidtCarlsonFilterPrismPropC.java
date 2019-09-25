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

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.filter.MeasurementModel;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import models.kalman.filter.DefaultProcessModel;
import models.kalman.filter.SchmidtCarlsonFilterPropC;
import parser.State;
import parser.VarList;
import parser.ast.Declaration;
import parser.ast.DeclarationInt;
import parser.ast.Expression;
import parser.type.Type;
import parser.type.TypeInt;
import prism.DefaultModelGenerator;
import prism.ModelType;
import prism.PrismException;
import prism.PrismLangException;

public class SchmidtCarlsonFilterPrismPropC extends DefaultModelGenerator {

	private State exploreState;
	private int max_time;
	private int s;
	private int t;
	private DefaultProcessModel pm;
	private MeasurementModel mm;
	private SchmidtCarlsonFilterPropC kf;
	private double nis;
	private int gLevel;
	private int decPlaces;
	private NormalDistribution sim_proc_noise, meas_noise;
	boolean observable, controllable;
	private RealMatrix G;
	private State state;
	private double sigma2;
	private double Dt;

	public SchmidtCarlsonFilterPrismPropC(int gLevel, int decPlaces, double sigma2, double Dt, RealMatrix G,
			NormalDistribution sim_proc_noise, NormalDistribution measNoise, DefaultProcessModel pm,
			MeasurementModel mm, int max_time) throws PrismException, IllegalArgumentException, IllegalStateException,
	InterruptedException, ExecutionException {
		this.pm = pm;
		this.mm = mm;
		this.G = G;
		this.gLevel = gLevel;
		this.sigma2 = sigma2;
		this.Dt = Dt;
		this.decPlaces = decPlaces;
		this.sim_proc_noise = sim_proc_noise;
		this.meas_noise = measNoise;
		this.kf = new SchmidtCarlsonFilterPropC(pm, mm, G, sigma2);
		this.max_time = max_time;
	}

	public void setMax_time(int max_time) {
		this.max_time = max_time;
	}

	@Override
	public ModelType getModelType() {
		return ModelType.DTMC;
	}

	@Override
	public int getNumVars() {
		return getVarNames().size();
	}

	@Override
	public List<String> getVarNames() {
		return Arrays.asList("s", // 0
				"t", // 1
				"z_s_int_s", // 2
				"z_s_frac_s", // 3
				"C_11_int_s", // 4
				"C_11_f_s", // 5
				"C_12_int_s", // 6
				"C_12_f_s", // 7
				"C_21_int_s", // 8
				"C_21_f_s", // 9
				"C_22_int_s", // 10
				"C_22_f_s", // 11
				"x_11_int_s", // 12
				"x_11_f_s", // 13
				"x_21_int_s", // 14
				"x_21_f_s", // 15
				"x_11_int_sim_s", // 16
				"x_11_f_sim_s", // 17
				"x_21_int_sim_s", // 18
				"x_21_f_sim_s", // 19
				"w_11_int_s", // 20
				"w_11_f_s", // 21
				"v_11_int_s", // 22
				"v_11_f_s", // 23
				"innov_11_int_s", // 24
				"innov_11_f_s", // 25
				"sinv_11_int_s", // 26
				"sinv_11_f_s", // 27
				"s_11_int_s", // 28
				"s_11_f_s", // 29
				"cond_11_int_s", // 30
				"cond_11_f_s"); // 31;
	}

	@Override
	public List<Type> getVarTypes() {
		return Arrays.asList(TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
				TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
				TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
				TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
				TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
				TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
				TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
				TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance());
	}

	@Override
	public int getNumLabels() {
		return 2;
	}

	@Override
	public List<String> getLabelNames() {
		return Arrays.asList("end", "isPD");
	}

	@Override
	public List<String> getRewardStructNames() {
		return Arrays.asList("cond");
	}

	@Override
	public State getInitialState() throws PrismException {
		state = new State(32);
		state.varValues[0] = 0; // s
		state.varValues[1] = 0; // t
		state.varValues[2] = 0; // z_s_int
		state.varValues[3] = 0; // z_s_frac
		state.varValues[4] = Computations.getIntegralPartAsInt(pm.getC().getEntry(0, 0), decPlaces);
		state.varValues[5] = Computations.getFractionalPartAsInt(pm.getC().getEntry(0, 0), decPlaces); // C_11
		state.varValues[6] = Computations.getIntegralPartAsInt(pm.getC().getEntry(0, 1), decPlaces);
		state.varValues[7] = Computations.getFractionalPartAsInt(pm.getC().getEntry(0, 1), decPlaces); // C_12
		state.varValues[8] = Computations.getIntegralPartAsInt(pm.getC().getEntry(1, 0), decPlaces);
		state.varValues[9] = Computations.getFractionalPartAsInt(pm.getC().getEntry(1, 0), decPlaces); // C_21
		state.varValues[10] = Computations.getIntegralPartAsInt(pm.getC().getEntry(1, 1), decPlaces);
		state.varValues[11] = Computations.getFractionalPartAsInt(pm.getC().getEntry(1, 1), decPlaces);// C_22
		state.varValues[12] = 0;
		state.varValues[13] = 0; // x_11
		state.varValues[14] = 0;
		state.varValues[15] = 0; // x_21

		state.varValues[16] = 0; // x11_int_sim_s
		state.varValues[17] = 0;
		state.varValues[18] = 0;
		state.varValues[19] = 0;

		state.varValues[20] = 0; // w_11_f_s;
		state.varValues[21] = 0;
		state.varValues[22] = 0; // v_11_f_s;
		state.varValues[23] = 0;

		state.varValues[24] = 0; // innov
		state.varValues[25] = 0;

		state.varValues[26] = 0; // sinv
		state.varValues[27] = 0;

		state.varValues[28] = 0; // s
		state.varValues[29] = 0;

		state.varValues[30] = 0; // cond number
		state.varValues[31] = 0;
		return state;
	}

	@Override
	public void exploreState(State exploreState) throws PrismException {
		this.exploreState = exploreState;
		s = ((Integer) exploreState.varValues[0]).intValue();
		t = ((Integer) exploreState.varValues[1]).intValue();
	}

	@Override
	public State getExploreState() {
		return exploreState;
	}

	@Override
	public int getNumChoices() throws PrismException {
		return 1;
	}

	@Override
	public double getTransitionProbability(int i, int offset) throws PrismException {

		double[] probs = Computations.calcTransProbAndExpectedVal(sim_proc_noise, gLevel, decPlaces);
		if (gLevel == 2) {
			return (t == max_time) ? 1.0 : (offset == 0 ? probs[0] : probs[2]);
		} else if (gLevel == 3) {
			if (t == max_time)
				return 1.0;
			else if (offset == 0)
				return probs[0];
			else if (offset == 1)
				return probs[2];
			else if (offset == 2)
				return probs[4];
		} else if (gLevel == 4) {
			if (t == max_time)
				return 1.0;
			else if (offset == 0)
				return probs[0];
			else if (offset == 1)
				return probs[2];
			else if (offset == 2)
				return probs[4];
			else if (offset == 3)
				return probs[6];
		} else if (gLevel == 5) {
			if (t == max_time)
				return 1.0;
			else if (offset == 0)
				return probs[0];
			else if (offset == 1)
				return probs[2];
			else if (offset == 2)
				return probs[4];
			else if (offset == 3)
				return probs[6];
			else if (offset == 4)
				return probs[8];
		} else if (gLevel == 6) {
			if (t == max_time)
				return 1.0;
			else if (offset == 0)
				return probs[0];
			else if (offset == 1)
				return probs[2];
			else if (offset == 2)
				return probs[4];
			else if (offset == 3)
				return probs[6];
			else if (offset == 4)
				return probs[8];
			else if (offset == 5)
				return probs[10];
		}
		return -1;
	}

	@Override
	public State computeTransitionTarget(int i, int offset) throws PrismException {

		State explore = getExploreState();

		RealVector x = new ArrayRealVector(new double[] {
				((int) explore.varValues[12] + (((int) explore.varValues[13] / Math.pow(10, decPlaces)))),
				((int) explore.varValues[14] + ((int) explore.varValues[15] / Math.pow(10, decPlaces))) });

		RealMatrix C = new Array2DRowRealMatrix(new double[][] {
			{ (int) explore.varValues[4] + ((int) explore.varValues[5] / Math.pow(10, decPlaces)),
				(int) explore.varValues[6] + ((int) explore.varValues[7] / Math.pow(10, decPlaces)) },
			{ (int) explore.varValues[8] + ((int) explore.varValues[9] / Math.pow(10, decPlaces)),
					(int) explore.varValues[10] + ((int) explore.varValues[11] / Math.pow(10, decPlaces)) } });

		RealVector xEntries = new ArrayRealVector(
				new double[] { (int) explore.varValues[16] + ((int) explore.varValues[17] / (Math.pow(10, decPlaces))),
						(int) explore.varValues[18] + ((int) explore.varValues[19] / Math.pow(10, decPlaces)) });

		// For discrete
		RealVector wEntries = new ArrayRealVector(new double[] {
				(int) explore.varValues[20]
						+ ((int) explore.varValues[21] / (Math.pow(10, decPlaces)) * (Math.pow(Dt, 2) / 2.)),
						(int) explore.varValues[20] + ((int) explore.varValues[21] / (Math.pow(10, decPlaces)) * Dt) });
		// System.out.println("xEntries is "+ xEntries);
		// System.out.println(xEntries);
		// System.out.println("wentries " + wEntries);
		// RealVector wEntries = new ArrayRealVector(new double[] {
		// (int) explore.varValues[20] + ((int) explore.varValues[21] / (Math.pow(10,
		// decPlaces))),
		// (int) explore.varValues[20] + ((int) explore.varValues[21] / (Math.pow(10,
		// decPlaces))) });
		// double wEntries =(int) explore.varValues[20] + ((int) explore.varValues[21] /
		// (Math.pow(10, decPlaces)));
		// RealVector GVector = new ArrayRealVector(
		// new double[] { (int) explore.varValues[20] + ((int) explore.varValues[21] /
		// (Math.pow(10, decPlaces))),
		// (int) explore.varValues[20] + ((int) explore.varValues[21] / (Math.pow(10,
		// decPlaces))) });

		RealVector x_sim = pm.getStateTransitionMatrix().operate(xEntries).add(wEntries);

		int x_11_int_sim_1 = Computations.getIntegralPartAsInt(x_sim.getEntry(0), decPlaces);
		int x_11_f_sim_1 = Computations.getFractionalPartAsInt(x_sim.getEntry(0), decPlaces);
		int x_21_int_sim_1 = Computations.getIntegralPartAsInt(x_sim.getEntry(1), decPlaces);
		int x_21_f_sim_1 = Computations.getFractionalPartAsInt(x_sim.getEntry(1), decPlaces);

		// Measurement emitted from simulation model
		RealVector vEntries = new ArrayRealVector(
				new double[] { (int) explore.varValues[22] + ((int) explore.varValues[23] / Math.pow(10, decPlaces)) });

		RealVector z_sim = mm.getMeasurementMatrix().operate(x_sim).add(vEntries);
		pm.setInitialStateEstimateVector(x);
		pm.setInitialErrorCovMatrix(C.multiply(C.transpose()));

		try {
			kf = new models.kalman.filter.SchmidtCarlsonFilterPropC(pm, mm, G, sigma2);
			kf.predict();
			kf.correct(z_sim);
		} catch (Exception e) {
		}

		int innov_int_11 = Computations.getIntegralPartAsInt(kf.getInnovation().getEntry(0), decPlaces);
		int innov_frac_11 = Computations.getFractionalPartAsInt(kf.getInnovation().getEntry(0), decPlaces);


		// -----------------------end of simulation
		// model-----------------------------------------------------------//

		int z_sim_int_1 = Computations.getIntegralPartAsInt(z_sim.getEntry(0), decPlaces);
		int z_sim_frac_1 = Computations.getFractionalPartAsInt(z_sim.getEntry(0), decPlaces);

		int x_post_11_int = Computations.getIntegralPartAsInt(kf.getStateEstimationVector().getEntry(0), decPlaces);
		int x_post_11_frac = Computations.getFractionalPartAsInt(kf.getStateEstimationVector().getEntry(0), decPlaces);

		int x_post_21_int = Computations.getIntegralPartAsInt(kf.getStateEstimationVector().getEntry(1), decPlaces);
		int x_post_21_frac = Computations.getFractionalPartAsInt(kf.getStateEstimationVector().getEntry(1), decPlaces);

		int C_post_11_int = Computations.getIntegralPartAsInt(kf.getC().getEntry(0, 0), decPlaces);
		int C_post_11_frac = Computations.getFractionalPartAsInt(kf.getC().getEntry(0, 0), decPlaces);
		int C_post_12_int = Computations.getIntegralPartAsInt(kf.getC().getEntry(0, 1), decPlaces);
		int C_post_12_frac = Computations.getFractionalPartAsInt(kf.getC().getEntry(0, 1), decPlaces);
		int C_post_21_int = Computations.getIntegralPartAsInt(kf.getC().getEntry(1, 0), decPlaces);
		int C_post_21_frac = Computations.getFractionalPartAsInt(kf.getC().getEntry(1, 0), decPlaces);
		int C_post_22_int = Computations.getIntegralPartAsInt(kf.getC().getEntry(1, 1), decPlaces);
		int C_post_22_frac = Computations.getFractionalPartAsInt(kf.getC().getEntry(1, 1), decPlaces);

		int cond_int = 0;
		int cond_frac = 0;

		State target = new State(exploreState);
		if (t == max_time) {
			return target;
		}

		else {
			if (s == 0 && t < max_time) {
				target.setValue(0, s + 1);
				target.setValue(1, t + 1);
				target.setValue(2, z_sim_int_1);
				target.setValue(3, z_sim_frac_1);
				target.setValue(16, 5);
				target.setValue(17, 0);
				target.setValue(18, 5);
				target.setValue(19, x_21_f_sim_1);
				target.setValue(20, Computations.getIntegralPartAsInt(
						Computations.calcTransProbAndExpectedVal(sim_proc_noise, gLevel, decPlaces)[2 * offset + 1],
						decPlaces));
				target.setValue(21, Computations.getFractionalPartAsInt(
						Computations.calcTransProbAndExpectedVal(sim_proc_noise, gLevel, decPlaces)[2 * offset + 1],
						decPlaces));
				target.setValue(22,
						Computations.getIntegralPartAsInt(
								Computations.calcTransProbAndExpectedVal(meas_noise, gLevel, decPlaces)[2 * offset + 1],
								decPlaces));
				target.setValue(23,
						Computations.getFractionalPartAsInt(
								Computations.calcTransProbAndExpectedVal(meas_noise, gLevel, decPlaces)[2 * offset + 1],
								decPlaces));
				target.setValue(24, innov_int_11);
				target.setValue(25, innov_frac_11);
				target.setValue(30, cond_int);
				target.setValue(31, cond_frac);
				return target;
			}

			if (s > 0) {
				target.setValue(0, s + 1);
				target.setValue(1, t + 1);
				target.setValue(2, z_sim_int_1);
				target.setValue(3, z_sim_frac_1);
				target.setValue(4, C_post_11_int); // P11
				target.setValue(5, C_post_11_frac); // P11
				target.setValue(6, C_post_12_int); // P12
				target.setValue(7, C_post_12_frac); // P12
				target.setValue(8, C_post_21_int);// P21
				target.setValue(9, C_post_21_frac);//
				target.setValue(10, C_post_22_int); // P22
				target.setValue(11, C_post_22_frac); // P22
				target.setValue(12, x_post_11_int); // x11
				target.setValue(13, x_post_11_frac); // x11
				target.setValue(14, x_post_21_int); // x21
				target.setValue(15, x_post_21_frac); // x21
				target.setValue(16, x_11_int_sim_1);
				target.setValue(17, x_11_f_sim_1);
				target.setValue(18, x_21_int_sim_1);
				target.setValue(19, x_21_f_sim_1);
				target.setValue(20, Computations.getIntegralPartAsInt(
						Computations.calcTransProbAndExpectedVal(sim_proc_noise, gLevel, decPlaces)[2 * offset + 1],
						decPlaces));
				target.setValue(21, Computations.getFractionalPartAsInt(
						Computations.calcTransProbAndExpectedVal(sim_proc_noise, gLevel, decPlaces)[2 * offset + 1],
						decPlaces));
				target.setValue(22,
						Computations.getIntegralPartAsInt(
								Computations.calcTransProbAndExpectedVal(meas_noise, gLevel, decPlaces)[2 * offset + 1],
								decPlaces));
				target.setValue(23,
						Computations.getFractionalPartAsInt(
								Computations.calcTransProbAndExpectedVal(meas_noise, gLevel, decPlaces)[2 * offset + 1],
								decPlaces));

				target.setValue(24, innov_int_11);
				target.setValue(25, innov_frac_11);
				target.setValue(30, cond_int);
				target.setValue(31, cond_frac);
				return target;
			}
			return null;

		}

	}

	@Override
	public boolean isLabelTrue(int i) throws PrismException {
		RealMatrix cPostPrism = new Array2DRowRealMatrix(
				new double[][] {
					{ (int) exploreState.varValues[4] + (int) exploreState.varValues[5] / Math.pow(10, decPlaces),
						(int) exploreState.varValues[6]
								+ (int) exploreState.varValues[7] / Math.pow(10, decPlaces) },
					{ (int) exploreState.varValues[8] + (int) exploreState.varValues[9] / Math.pow(10, decPlaces),
									(int) exploreState.varValues[10]
											+ (int) exploreState.varValues[11] / Math.pow(10, decPlaces) } });

		EigenDecomposition eig = new EigenDecomposition(cPostPrism);

		double[] eigValues = eig.getRealEigenvalues();

		boolean isPD = eigValues[0] > 0 && eigValues[1] > 0;

		if (i == 0) {

			return t == max_time;
		} else if (i == 1) {
			return  isPD;
		}

		return false;
	}

	// state variables
	@Override
	public VarList createVarList() {

		VarList varList = new VarList();
		try {
			varList.addVar(
					new Declaration("s", new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0,
					null);
			varList.addVar(new Declaration("t", new DeclarationInt(Expression.Int(0), Expression.Int(max_time))), 0,
					null);
			varList.addVar(new Declaration("z_s_int_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("z_s_f_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("C_11_int_s",
					new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("C_11_int_f_s",
					new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("C_12_int_s",
					new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("C_12_int_f_s",
					new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("C_21_int_s",
					new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("C_21_int_f_s",
					new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("C_22_int_s",
					new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("C_22_int_f_s",
					new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("x_11_int_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("x_11_int_f_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("x_21_int_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("x_21_int_f_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("x_11_int_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("x_11_int_sim_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("x_11_f_sim_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("x_21_int_sim_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("x_21_f_sim_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("w_11_int_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("w_11_f_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("v_11_int_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("v_11_f_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("innov_11_int_s",
					new DeclarationInt(Expression.Int(-100000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("innov_11_f_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("sinv_11_int_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("sinv_11_f_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("s_11_int_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("s_11_f_s",
					new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("cond_11_int_s",
					new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
			varList.addVar(new Declaration("cond_11_f_s",
					new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
		} catch (PrismLangException e) {
		}

		return varList;
	}

	@Override
	public int getNumTransitions(int i) throws PrismException {
		if (gLevel == 2)
			return (t == max_time) ? 1 : 2;
		else if (gLevel == 3)
			return (t == max_time) ? 1 : 4;
		else if (gLevel == 33)
			return (t == max_time) ? 1 : 3;
		else if (gLevel == 4)
			return (t == max_time) ? 1 : 6;
		else if (gLevel == 44)
			return (t == max_time) ? 1 : 5;
		else if (gLevel == 5)
			return (t == max_time) ? 1 : 5;
		else
			return -1;
	}

	public void setNis(double nis) {
		this.nis = nis;
	}

	public double getNis() {
		return nis;
	}

	@Override
	public double getStateReward(int r, State state) throws PrismException {
		RealMatrix aPostCovMatrixPRISM = new Array2DRowRealMatrix(new double[][] {
			{ (int) state.varValues[4] + (int) state.varValues[5] / Math.pow(10, decPlaces),
				(int) state.varValues[6] + (int) state.varValues[7] / Math.pow(10, decPlaces) },
			{ (int) state.varValues[8] + (int) state.varValues[9] / Math.pow(10, decPlaces),
					(int) state.varValues[10] + (int) state.varValues[11] / Math.pow(10, decPlaces) } });


		SingularValueDecomposition svd = new SingularValueDecomposition(aPostCovMatrixPRISM);
		double cond_number_prism = svd.getConditionNumber();
		if (r == 0 && (int) state.varValues[0] >= 0) {
			return cond_number_prism;
		}
		return 0;
	}

	@Override
	public Object getTransitionAction(int i) throws PrismException {
		return null;
	}

	@Override
	public Object getTransitionAction(int i, int offset) throws PrismException {
		return null;
	}

	@Override
	public double getStateActionReward(int r, State state, Object action) throws PrismException {
		return 0.0;
	}

	@Override
	public boolean rewardStructHasTransitionRewards(int i) {
		return false;
	}

}
