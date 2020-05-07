/**
 * Copyright (C) 2018-2020 Alexandros Evangelidis.
 * 
 * VerFilter
 * 
 * This file is part of VerFilter.
 * 
 * VerFilter is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * VerFilter is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program. If
 * not, see <http://www.gnu.org/licenses/>.
 */


import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.util.Precision;
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

public class UDFilterPrismPropUD extends DefaultModelGenerator {

  private State exploreState;
  private int max_time;
  private int s;
  private int t;
  private DefaultProcessModel pm;
  private MeasurementModel mm;
  private UDFilterPropUD kf;
  private double nis;
  private int gLevel;
  private int decPlaces;
  private NormalDistribution sim_proc_noise, meas_noise;
  boolean observable, controllable;
  private RealMatrix G;
  private State state;
  double condNumber;
  private double sigma2;
  private double Dt;

  public UDFilterPrismPropUD(int gLevel, int decPlaces, double sigma2, double Dt, RealMatrix G,
      NormalDistribution sim_proc_noise, NormalDistribution measNoise, DefaultProcessModel pm,
      MeasurementModel mm, int max_time) throws PrismException, IllegalArgumentException,
      IllegalStateException, InterruptedException, ExecutionException {
    this.pm = pm;
    this.mm = mm;
    this.G = G;
    this.gLevel = gLevel;
    this.sigma2 = sigma2;
    this.Dt = Dt;
    this.decPlaces = decPlaces;
    this.sim_proc_noise = sim_proc_noise;
    this.meas_noise = measNoise;
    this.kf = new UDFilterPropUD(pm, mm, G, sigma2);
    this.max_time = max_time;
  }

  public void setMax_time(int max_time) {
    this.max_time = max_time;
  }

  // The model is a discrete-time Markov chain (DTMC)
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
        "U_11_int_s", // 4
        "U_11_f_s", // 5
        "U_12_int_s", // 6
        "U_12_f_s", // 7
        "U_21_int_s", // 8
        "U_21_f_s", // 9
        "U_22_int_s", // 10
        "U_22_f_s", // 11
        "D_11_int_s", // 12
        "D_11_f_s", // 13
        "D_12_int_s", // 14
        "D_12_f_s", // 15
        "D_21_int_s", // 16
        "D_21_f_s", // 17
        "D_22_int_s", // 18
        "D_22_f_s", // 19
        "x_11_int_s", // 20
        "x_11_f_s", // 21
        "x_21_int_s", // 22
        "x_21_f_s", // 23
        "x_11_int_sim_s", // 24
        "x_11_f_sim_s", // 25
        "x_21_int_sim_s", // 26
        "x_21_f_sim_s", // 27
        "w_11_int_s", // 28
        "w_11_f_s", // 29
        "v_11_int_s", // 30
        "v_11_f_s", // 31
        "innov_11_int_s", // 32
        "innov_11_f_s", // 33
        "sinv_11_int_s", // 34
        "sinv_11_f_s", // 35
        "s_11_int_s", // 36
        "s_11_f_s", // 37
        "cond_11_int_s", // 38
        "cond_11_f_s"); // 39;
  }

  // checked on 28/05/2019
  @Override
  public List<Type> getVarTypes() {
    return Arrays.asList(TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
        TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
        TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
        TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
        TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
        TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
        TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
        TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
        TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
        TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(), TypeInt.getInstance(),
        TypeInt.getInstance());
  }

  @Override
  public int getNumLabels() {
    return 9;
  }

  @Override
  public List<String> getLabelNames() {
    return Arrays.asList("end", "cond_bound_sp", "inv_cond_in_range", "inv_cond_one", "isSymmetric",
        "isPSD", "isPD", "isID", "isNV");
  }

  @Override
  public List<String> getRewardStructNames() {

    // return Arrays.asList("nis");
    return Arrays.asList("cond", "inv_cond", "nis", "inRange", "nis_avg", "log_cond_bound_float",
        "dp_error_est");
  }

  @Override
  public State getInitialState() throws PrismException {
    state = new State(40);
    state.varValues[0] = 0; // s
    state.varValues[1] = 0; // t
    state.varValues[2] = 0; // z_s_int
    state.varValues[3] = 0; // z_s_frac
    state.varValues[4] = Computations.getIntegralPartAsInt(pm.getU().getEntry(0, 0), decPlaces);
    state.varValues[5] = Computations.getFractionalPartAsInt(pm.getU().getEntry(0, 0), decPlaces); // U_11
    state.varValues[6] = Computations.getIntegralPartAsInt(pm.getU().getEntry(0, 1), decPlaces);
    state.varValues[7] = Computations.getFractionalPartAsInt(pm.getU().getEntry(0, 1), decPlaces); // U_12
    state.varValues[8] = Computations.getIntegralPartAsInt(pm.getU().getEntry(1, 0), decPlaces);
    state.varValues[9] = Computations.getFractionalPartAsInt(pm.getU().getEntry(1, 0), decPlaces); // U_21
    state.varValues[10] = Computations.getIntegralPartAsInt(pm.getU().getEntry(1, 1), decPlaces);
    state.varValues[11] = Computations.getFractionalPartAsInt(pm.getU().getEntry(1, 1), decPlaces);// U_22
    state.varValues[12] = Computations.getIntegralPartAsInt(pm.getD().getEntry(0, 0), decPlaces);
    state.varValues[13] = Computations.getFractionalPartAsInt(pm.getD().getEntry(0, 0), decPlaces); // D_11
    state.varValues[14] = Computations.getIntegralPartAsInt(pm.getD().getEntry(0, 1), decPlaces);
    state.varValues[15] = Computations.getFractionalPartAsInt(pm.getD().getEntry(0, 1), decPlaces); // D_12
    state.varValues[16] = Computations.getIntegralPartAsInt(pm.getD().getEntry(1, 0), decPlaces);
    state.varValues[17] = Computations.getFractionalPartAsInt(pm.getD().getEntry(1, 0), decPlaces); // D_21
    state.varValues[18] = Computations.getIntegralPartAsInt(pm.getD().getEntry(1, 1), decPlaces);
    state.varValues[19] = Computations.getFractionalPartAsInt(pm.getD().getEntry(1, 1), decPlaces);// D_22
    state.varValues[20] = 0;
    state.varValues[21] = 0; // x_11
    state.varValues[22] = 0;
    state.varValues[23] = 0; // x_21

    state.varValues[24] = 0; // x11_int_sim_s
    state.varValues[25] = 0;
    state.varValues[26] = 0;
    state.varValues[27] = 0;

    state.varValues[28] = 0; // w_11_int_s;
    state.varValues[29] = 0;
    state.varValues[30] = 0; // v_11_f_s;
    state.varValues[31] = 0;

    state.varValues[32] = 0; // innov
    state.varValues[33] = 0;

    state.varValues[34] = 0; // sinv
    state.varValues[35] = 0;

    state.varValues[36] = 0; // s
    state.varValues[37] = 0;

    state.varValues[38] = 0; // cond number
    state.varValues[39] = 0;
    // System.out.println("INITIAL STATE = " + state);
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
    // double[] probs = {0.02,0 , 0.14, 0.69, 0.69, 0.05, .13,0.06,0.02};
    // System.out.println("sum is "+ gLevel);
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
      else if (offset == 3)
        return probs[6];
    } else if (gLevel == 33) {
      if (t == max_time)
        return 1.0;
      else if (offset == 0)
        return probs[0];
      else if (offset == 1)
        return probs[2];
      else if (offset == 2)
        return probs[4];
    }

    else if (gLevel == 4) {
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
    } else if (gLevel == 44) {
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
    } else if (gLevel == 5) {
      // System.out.println("check");

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

    }
    return -1;
  }

  @Override
  public State computeTransitionTarget(int i, int offset) throws PrismException {
    // System.out.println("CONTROLLABLE "+ controllable);
    // if (controllable) {
    State explore = getExploreState();
    // System.out.println("------------------------------------");
    // System.out.println("explorestate is " + explore);

    // estimated kalman filter states

    RealVector x = new ArrayRealVector(new double[] {
        ((int) explore.varValues[20] + (((int) explore.varValues[21] / Math.pow(10, decPlaces)))),
        ((int) explore.varValues[22] + ((int) explore.varValues[23] / Math.pow(10, decPlaces)))});
    // System.out.println("x11 frac is "+ explore.varValues[13]);
    // System.out.println("x is " + x);
    RealMatrix U =
        new Array2DRowRealMatrix(new double[][] {
            {(int) explore.varValues[4] + ((int) explore.varValues[5] / Math.pow(10, decPlaces)),
                (int) explore.varValues[6]
                    + ((int) explore.varValues[7] / Math.pow(10, decPlaces))},
            {(int) explore.varValues[8] + ((int) explore.varValues[9] / Math.pow(10, decPlaces)),
                (int) explore.varValues[10]
                    + ((int) explore.varValues[11] / Math.pow(10, decPlaces))}});

    RealMatrix D =
        new Array2DRowRealMatrix(new double[][] {
            {(int) explore.varValues[12] + ((int) explore.varValues[13] / Math.pow(10, decPlaces)),
                (int) explore.varValues[14]
                    + ((int) explore.varValues[15] / Math.pow(10, decPlaces))},
            {(int) explore.varValues[16] + ((int) explore.varValues[17] / Math.pow(10, decPlaces)),
                (int) explore.varValues[18]
                    + ((int) explore.varValues[19] / Math.pow(10, decPlaces))}});

    // System.out.println("P is"+ kf.getErrorCovarianceMatrix());
    //
    // System.out.println("cond_int is " + cond_int);
    // System.out.println("cond frac is " + cond_frac);
    // System.out.println("COND NUMBER as double :" + condNumber);
    // System.out.println("COND NUMBER :" + (cond_int + (cond_frac / Math.pow(10,
    // decPlaces))));
    // System.out.println("Condition number from PRISM loop: "
    // + ((int) explore.varValues[30] + ((int) explore.varValues[31] / (Math.pow(10,
    // decPlaces)))));

    // x_11_int_sim_s etc
    RealVector xEntries = new ArrayRealVector(new double[] {
        (int) explore.varValues[24] + ((int) explore.varValues[25] / (Math.pow(10, decPlaces))),
        (int) explore.varValues[26] + ((int) explore.varValues[27] / Math.pow(10, decPlaces))});
    // System.out.println(xEntries);

    // For discrete
    RealVector wEntries = new ArrayRealVector(new double[] {
        (int) explore.varValues[28]
            + ((int) explore.varValues[29] / (Math.pow(10, decPlaces)) * (Math.pow(Dt, 2) / 2.)),
        (int) explore.varValues[28]
            + ((int) explore.varValues[29] / (Math.pow(10, decPlaces)) * Dt)});

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

    // System.out.println("wentries: " + wEntries);
    RealVector x_sim = pm.getStateTransitionMatrix().operate(xEntries).add(wEntries);
    // System.out.println("xEntries is from PRISM states " + xEntries);
    // System.out.println("xsim is the vector NONPRISM " + x_sim);

    int x_11_int_sim_1 = Computations.getIntegralPartAsInt(x_sim.getEntry(0), decPlaces);
    int x_11_f_sim_1 = Computations.getFractionalPartAsInt(x_sim.getEntry(0), decPlaces);
    int x_21_int_sim_1 = Computations.getIntegralPartAsInt(x_sim.getEntry(1), decPlaces);
    int x_21_f_sim_1 = Computations.getFractionalPartAsInt(x_sim.getEntry(1), decPlaces);

    // Measurement emitted from simulation model
    // Maybe add randomness here????
    RealVector vEntries = new ArrayRealVector(new double[] {
        (int) explore.varValues[30] + ((int) explore.varValues[31] / Math.pow(10, decPlaces))});

    // System.out.println("v entries is " + vEntries);
    RealVector z_sim = mm.getMeasurementMatrix().operate(x_sim).add(vEntries);
    // System.out.println("measurement is " + z_sim);
    pm.setInitialStateEstimateVector(x);
    pm.setInitialErrorCovMatrix(U.multiply(D).multiply(U.transpose()));

    try {
      // need KF objects for each path
      kf = new UDFilterPropUD(pm, mm, G, sigma2);
      kf.predict();
      kf.correct(z_sim);
    } catch (Exception e) {
    }

    // System.out.println("x_pred_entries are "+ x);

    // System.out.println("a posteriori is " + kf.getStateEstimationVector() + " "
    // + pm.getInitialStateEstimateVector());
    // System.out.println(" a priori cov is " + kf.getErrorCovarianceMatrix());
    // Simulation model ;
    // First the process

    // System.out.println("INNOVATION IS " + kf.getInnovation() + "s_inv is" +
    // kf.getInnovationCovarianceInv());
    // System.out.println("innovation " + kf.getInnovation().getEntry(0));
    int innov_int_11 = Computations.getIntegralPartAsInt(kf.getInnovation().getEntry(0), decPlaces);

    int innov_frac_11 =
        Computations.getFractionalPartAsInt(kf.getInnovation().getEntry(0), decPlaces);
    // System.out.println("innovation " + (innov_int_11 + (innov_frac_11 / Math.pow(10,
    // decPlaces))));
    // int sinv_int_11 =
    // Computations.getIntegralPartAsInt(kf.getInnovationCovarianceInv().getEntry(0,
    // 0), decPlaces);
    // int sinv_frac_11 =
    // Computations.getFractionalPartAsInt(kf.getInnovationCovarianceInv().getEntry(0,
    // 0),
    // decPlaces);

    // int s_int_11 =
    // Computations.getIntegralPartAsInt(kf.getInnovationCovariance().getEntry(0,
    // 0), decPlaces);
    // int s_frac_11 =
    // Computations.getFractionalPartAsInt(kf.getInnovationCovariance().getEntry(0,
    // 0), decPlaces);

    // -----------------------end of simulation
    // model-----------------------------------------------------------//
    // System.out.println("z_sim get entry " + z_sim.getEntry(0));

    int z_sim_int_1 = Computations.getIntegralPartAsInt(z_sim.getEntry(0), decPlaces);
    int z_sim_frac_1 = Computations.getFractionalPartAsInt(z_sim.getEntry(0), decPlaces);
    // System.out.println("fractional part of z_Sim_frac_ " + z_sim_frac_1);

    int x_post_11_int =
        Computations.getIntegralPartAsInt(kf.getStateEstimationVector().getEntry(0), decPlaces);
    int x_post_11_frac =
        Computations.getFractionalPartAsInt(kf.getStateEstimationVector().getEntry(0), decPlaces);

    int x_post_21_int =
        Computations.getIntegralPartAsInt(kf.getStateEstimationVector().getEntry(1), decPlaces);
    int x_post_21_frac =
        Computations.getFractionalPartAsInt(kf.getStateEstimationVector().getEntry(1), decPlaces);

    RealVector aPostStateEstimatePrism =
        new ArrayRealVector(new double[] {x_post_11_int + x_post_11_frac / Math.pow(10, decPlaces),
            x_post_21_int + x_post_21_frac / Math.pow(10, decPlaces)});
    // System.out.println("PRISM: a post state estimate: " + aPostStateEstimatePrism);
    // System.out.println(aPostStateEstimatePrism);
    //
    // RealMatrix P = new Array2DRowRealMatrix(new double[][] {
    // { (int) explore.varValues[4] + ((int) explore.varValues[5] / Math.pow(10,
    // decPlaces)),
    // (int) explore.varValues[6] + ((int) explore.varValues[7] / Math.pow(10,
    // decPlaces)) },
    // { (int) explore.varValues[8] + ((int) explore.varValues[9] / Math.pow(10,
    // decPlaces)),
    // (int) explore.varValues[10] + ((int) explore.varValues[11] / Math.pow(10,
    // decPlaces)) } });

    // System.out.println("error cov: "+ kf.getErrorCovarianceMatrix());

    int U_post_11_int = Computations.getIntegralPartAsInt(kf.getU().getEntry(0, 0), decPlaces);
    int U_post_11_frac = Computations.getFractionalPartAsInt(kf.getU().getEntry(0, 0), decPlaces);
    int U_post_12_int = Computations.getIntegralPartAsInt(kf.getU().getEntry(0, 1), decPlaces);
    int U_post_12_frac = Computations.getFractionalPartAsInt(kf.getU().getEntry(0, 1), decPlaces);
    int U_post_21_int = Computations.getIntegralPartAsInt(kf.getU().getEntry(1, 0), decPlaces);
    int U_post_21_frac = Computations.getFractionalPartAsInt(kf.getU().getEntry(1, 0), decPlaces);
    int U_post_22_int = Computations.getIntegralPartAsInt(kf.getU().getEntry(1, 1), decPlaces);
    int U_post_22_frac = Computations.getFractionalPartAsInt(kf.getU().getEntry(1, 1), decPlaces);

    int D_post_11_int = Computations.getIntegralPartAsInt(kf.getD().getEntry(0, 0), decPlaces);
    int D_post_11_frac = Computations.getFractionalPartAsInt(kf.getD().getEntry(0, 0), decPlaces);
    int D_post_12_int = Computations.getIntegralPartAsInt(kf.getD().getEntry(0, 1), decPlaces);
    int D_post_12_frac = Computations.getFractionalPartAsInt(kf.getD().getEntry(0, 1), decPlaces);
    int D_post_21_int = Computations.getIntegralPartAsInt(kf.getD().getEntry(1, 0), decPlaces);
    int D_post_21_frac = Computations.getFractionalPartAsInt(kf.getD().getEntry(1, 0), decPlaces);
    int D_post_22_int = Computations.getIntegralPartAsInt(kf.getD().getEntry(1, 1), decPlaces);
    int D_post_22_frac = Computations.getFractionalPartAsInt(kf.getD().getEntry(1, 1), decPlaces);

    //
    RealMatrix Upost = new Array2DRowRealMatrix(new double[][] {
        {U_post_11_int + U_post_11_frac / Math.pow(10, decPlaces),
            U_post_12_int + U_post_12_frac / Math.pow(10, decPlaces)},
        {U_post_21_int + U_post_21_frac / Math.pow(10, decPlaces),
            U_post_22_int + U_post_22_frac / Math.pow(10, decPlaces)}});

    RealMatrix Dpost = new Array2DRowRealMatrix(new double[][] {
        {D_post_11_int + D_post_11_frac / Math.pow(10, decPlaces),
            D_post_12_int + D_post_12_frac / Math.pow(10, decPlaces)},
        {D_post_21_int + D_post_21_frac / Math.pow(10, decPlaces),
            D_post_22_int + D_post_22_frac / Math.pow(10, decPlaces)}});

    RealMatrix aPostCovMatrixPRISM = Upost.multiply(Dpost).multiply(Upost.transpose());
    // System.out.println("PRISM: A post cov matrix" + aPostCovMatrixPRISM);

    // //
    // SingularValueDecomposition svd2 = new
    // SingularValueDecomposition(aPostCovMatrixPRISM);

    // System.out.println("Inside loop condition number " +
    // svd2.getConditionNumber());
    // SingularValueDecomposition svd = new SingularValueDecomposition(P);
    // System.out.println("Inside loop condition number " +
    // svd.getConditionNumber());
    // System.out.println("Matrix P inside loop " + P);
    // condNumber = svd2.getConditionNumber();

    int cond_int = Computations.getIntegralPartAsInt(condNumber, decPlaces);
    int cond_frac = Computations.getFractionalPartAsInt(condNumber, decPlaces);
    // System.out.println("A-posteriori error cov matrix PRISM: " +
    // aPostCovMatrixPRISM);
    State target = new State(exploreState);
    if (t == max_time) {
      return target;
    }
    // "t", // 1
    // "z_s_int_s", // 2
    // "z_s_frac_s", // 3
    // "C_11_int_s", // 4
    // "C_11_f_s", // 5
    // "C_12_int_s", // 6
    // "C_12_f_s", // 7
    // "C_21_int_s", // 8
    // "C_21_f_s", // 9
    // "C_22_int_s", // 10
    // "C_22_f_s", // 11
    // "x_11_int_s", // 12
    // "x_11_f_s", // 13
    // "x_21_int_s", // 14
    // "x_21_f_s", // 15
    // "x_11_int_sim_s", // 16
    // "x_11_f_sim_s", // 17
    // "x_21_int_sim_s", // 18
    // "x_21_f_sim_s", // 19
    // "w_11_int_s", // 20
    // "w_11_f_s", // 21
    // "v_11_int_s", // 22
    // "v_11_f_s", // 23
    // "innov_11_int_s", // 24
    // "innov_11_f_s", // 25
    // "sinv_11_int_s", // 26
    // "sinv_11_f_s", // 27
    // "s_11_int_s", // 28
    // "s_11_f_s", // 29
    // "cond_11_int_s", // 30
    // "cond_11_f_s"); // 31;
    else {
      if (s == 0 && t < max_time) {
        target.setValue(0, s + 1);
        target.setValue(1, t + 1);
        target.setValue(2, z_sim_int_1);
        target.setValue(3, z_sim_frac_1);
        target.setValue(24, 5);
        target.setValue(25, 0);
        target.setValue(26, 5);
        target.setValue(27, x_21_f_sim_1);
        // System.out.println("offset is "+ offset);
        target.setValue(28,
            Computations.getIntegralPartAsInt(Computations
                .calcTransProbAndExpectedVal(sim_proc_noise, gLevel, decPlaces)[2 * offset + 1],
                decPlaces));
        target.setValue(29,
            Computations.getFractionalPartAsInt(Computations
                .calcTransProbAndExpectedVal(sim_proc_noise, gLevel, decPlaces)[2 * offset + 1],
                decPlaces));
        target.setValue(30, Computations.getIntegralPartAsInt(
            Computations.calcTransProbAndExpectedVal(meas_noise, gLevel, decPlaces)[2 * offset + 1],
            decPlaces));
        target.setValue(31, Computations.getFractionalPartAsInt(
            Computations.calcTransProbAndExpectedVal(meas_noise, gLevel, decPlaces)[2 * offset + 1],
            decPlaces));

        target.setValue(32, innov_int_11);
        target.setValue(33, innov_frac_11);
        target.setValue(34, cond_int);
        target.setValue(35, cond_frac);
        return target;
      }

      if (s > 0) {
        target.setValue(0, s + 1);
        target.setValue(1, t + 1);
        target.setValue(2, z_sim_int_1);
        target.setValue(3, z_sim_frac_1);
        target.setValue(4, U_post_11_int); // P11
        target.setValue(5, U_post_11_frac); // P11
        target.setValue(6, U_post_12_int); // P12
        target.setValue(7, U_post_12_frac); // P12
        target.setValue(8, U_post_21_int);// P21
        target.setValue(9, U_post_21_frac);//
        target.setValue(10, U_post_22_int); // P22
        target.setValue(11, U_post_22_frac); // P22
        target.setValue(12, D_post_11_int); // P11
        target.setValue(13, D_post_11_frac); // P11
        target.setValue(14, D_post_12_int); // P12
        target.setValue(15, D_post_12_frac); // P12
        target.setValue(16, D_post_21_int);// P21
        target.setValue(17, D_post_21_frac);//
        target.setValue(18, D_post_22_int); // P22
        target.setValue(19, D_post_22_frac); // P22
        target.setValue(20, x_post_11_int); // x11
        target.setValue(21, x_post_11_frac); // x11
        target.setValue(22, x_post_21_int); // x21
        target.setValue(23, x_post_21_frac); // x21
        target.setValue(24, x_11_int_sim_1);
        target.setValue(25, x_11_f_sim_1);
        target.setValue(26, x_21_int_sim_1);
        target.setValue(27, x_21_f_sim_1);
        target.setValue(28,
            Computations.getIntegralPartAsInt(Computations
                .calcTransProbAndExpectedVal(sim_proc_noise, gLevel, decPlaces)[2 * offset + 1],
                decPlaces));
        target.setValue(29,
            Computations.getFractionalPartAsInt(Computations
                .calcTransProbAndExpectedVal(sim_proc_noise, gLevel, decPlaces)[2 * offset + 1],
                decPlaces));
        target.setValue(30, Computations.getIntegralPartAsInt(
            Computations.calcTransProbAndExpectedVal(meas_noise, gLevel, decPlaces)[2 * offset + 1],
            decPlaces));
        target.setValue(31, Computations.getFractionalPartAsInt(
            Computations.calcTransProbAndExpectedVal(meas_noise, gLevel, decPlaces)[2 * offset + 1],
            decPlaces));

        target.setValue(32, innov_int_11);
        target.setValue(33, innov_frac_11);
        // target.setValue(26, sinv_int_11);
        // target.setValue(27, sinv_frac_11);
        // target.setValue(28, s_int_11);
        // target.setValue(29, s_frac_11);
        target.setValue(34, cond_int);
        target.setValue(35, cond_frac);
        return target;
      }

      return null;
    }

  }

  @Override
  public boolean isLabelTrue(int i) throws PrismException {
    RealMatrix DPostPrism = new Array2DRowRealMatrix(new double[][] {
        {(int) exploreState.varValues[12]
            + (int) exploreState.varValues[13] / Math.pow(10, decPlaces),
            (int) exploreState.varValues[14]
                + (int) exploreState.varValues[15] / Math.pow(10, decPlaces)},
        {(int) exploreState.varValues[16]
            + (int) exploreState.varValues[17] / Math.pow(10, decPlaces),
            (int) exploreState.varValues[18]
                + (int) exploreState.varValues[19] / Math.pow(10, decPlaces)}});


    RealMatrix UPostPrism = new Array2DRowRealMatrix(new double[][] {{
        (int) exploreState.varValues[4] + (int) exploreState.varValues[5] / Math.pow(10, decPlaces),
        (int) exploreState.varValues[6]
            + (int) exploreState.varValues[7] / Math.pow(10, decPlaces)},
        {(int) exploreState.varValues[8]
            + (int) exploreState.varValues[9] / Math.pow(10, decPlaces),
            (int) exploreState.varValues[10]
                + (int) exploreState.varValues[11] / Math.pow(10, decPlaces)}});


    RealMatrix P_recons = UPostPrism.multiply(DPostPrism).multiply(UPostPrism.transpose());

    // System.out.println(P_recons);

    // SingularValueDecomposition svd = new SingularValueDecomposition(cPostPrism);
    EigenDecomposition eig = new EigenDecomposition(P_recons);
    // double[] singValues = svd.getSingularValues();
    double[] eigValues = eig.getRealEigenvalues();

    // double cond_number = svd.getConditionNumber();
    // boolean cond_bound_sp = Math.log10(cond_number) > 6;
    // boolean cond_bound_dp = Math.log10(cond_number) > 15;
    // double inverse_cond_number = svd.getInverseConditionNumber();

    // boolean inv_cond_in_range = (int) exploreState.varValues[0] > 0 &&
    // inverse_cond_number >= 0.9
    // && inverse_cond_number <= 1.0;
    // boolean inv_cond_one = inverse_cond_number == 1.0;
    // boolean isDiagPos = cPostPrism.getEntry(1, 0) > 0 && cPostPrism.getEntry(1,
    // 0) > 0;

    // boolean isSymmetric = MatrixUtils.isSymmetric(cPostPrism, Precision.EPSILON);

    // boolean isPSD = eigValues[0] == 0 || eigValues[1] == 0;
    boolean isPD = eigValues[0] > 0 && eigValues[1] > 0;
    // boolean isID = (eigValues[0] > 0 && eigValues[1] < 0) || (eigValues[1] > 0 &&
    // eigValues[0] < 0);

    // boolean isNV = (cPostPrism.getEntry(0, 0) < 0) || (cPostPrism.getEntry(0, 1)
    // < 0)
    // || (cPostPrism.getEntry(1, 0) < 0) || (cPostPrism.getEntry(1, 1) < 0);

    // Get the diagonal entries of D
    double diagElemOne = DPostPrism.getEntry(0, 0);
    double diagElemTwo = DPostPrism.getEntry(1, 1);
    // boolean isPD = diagElemOne > 0 && diagElemTwo > 0;
    if (i == 0) {
      return t == max_time;
    } else if (i == 1) {
      // System.out.println("is label true " + cPostPrism);
      // System.out.println("Inside label cond number" + cond_number);
      // System.out.println("Inside label "+ Math.log10(cond_number));
      // return cond_bound_dp;
      return true;
    }

    else if (i == 2) {
      // System.out.println("Inside label " + Math.log10(cond_number));
      // return inv_cond_in_range;
      return true;
    }

    else if (i == 3) {
      // System.out.println("Inside label "+ Math.log10(cond_number));
      // return inv_cond_one;
      return true;
    }

    else if (i == 4) {
      // return isSymmetric;
      return true;
    }

    else if (i == 5) {
      return true;
    }

    else if (i == 6) {
      // System.out.println("eig 1: "+eigValues[0] + " eig 2:"+ eigValues[1]);
      return isPD;
    }

    else if (i == 7) {
      // return isID;
    }

    else if (i == 8) {
      // return isNV;
    }
    return false;
  }

  // state variables
  @Override
  public VarList createVarList() {

    VarList varList = new VarList();
    try {
      varList.addVar(
          new Declaration("s", new DeclarationInt(Expression.Int(0), Expression.Int(150))), 0,
          null);
      varList.addVar(
          new Declaration("t", new DeclarationInt(Expression.Int(0), Expression.Int(max_time))), 0,
          null);
      varList.addVar(
          new Declaration("z_s_int_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("z_s_f_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(new Declaration("U_11_int_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("U_11_int_f_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("U_12_int_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("U_12_int_f_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("U_21_int_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("U_21_int_f_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("U_22_int_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("U_22_int_f_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("D_11_int_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("D_11_int_f_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("D_12_int_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("D_12_int_f_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("D_21_int_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("D_21_int_f_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("D_22_int_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(new Declaration("D_22_int_f_s",
          new DeclarationInt(Expression.Int(0), Expression.Int(Integer.MAX_VALUE))), 0, null);
      varList.addVar(
          new Declaration("x_11_int_s",
              new DeclarationInt(Expression.Int(-1000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("x_11_int_f_s",
              new DeclarationInt(Expression.Int(-1000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("x_21_int_s",
              new DeclarationInt(Expression.Int(-1000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("x_21_int_f_s",
              new DeclarationInt(Expression.Int(-1000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("x_11_int_s",
              new DeclarationInt(Expression.Int(-500), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("x_11_int_sim_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("x_11_f_sim_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("x_21_int_sim_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("x_21_f_sim_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("w_11_int_s",
              new DeclarationInt(Expression.Int(-1000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("w_11_f_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("v_11_int_s",
              new DeclarationInt(Expression.Int(-500), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("v_11_f_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("innov_11_int_s",
              new DeclarationInt(Expression.Int(-100000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("innov_11_f_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("sinv_11_int_s",
              new DeclarationInt(Expression.Int(-100000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("sinv_11_f_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("s_11_int_s",
              new DeclarationInt(Expression.Int(-100000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
      varList.addVar(
          new Declaration("s_11_f_s",
              new DeclarationInt(Expression.Int(-10000), Expression.Int(Integer.MAX_VALUE))),
          0, null);
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

    RealMatrix DPostPrism = new Array2DRowRealMatrix(new double[][] {
        {(int) exploreState.varValues[12]
            + (int) exploreState.varValues[13] / Math.pow(10, decPlaces),
            (int) exploreState.varValues[14]
                + (int) exploreState.varValues[15] / Math.pow(10, decPlaces)},
        {(int) exploreState.varValues[16]
            + (int) exploreState.varValues[17] / Math.pow(10, decPlaces),
            (int) exploreState.varValues[18]
                + (int) exploreState.varValues[19] / Math.pow(10, decPlaces)}});


    RealMatrix UPostPrism = new Array2DRowRealMatrix(new double[][] {{
        (int) exploreState.varValues[4] + (int) exploreState.varValues[5] / Math.pow(10, decPlaces),
        (int) exploreState.varValues[6]
            + (int) exploreState.varValues[7] / Math.pow(10, decPlaces)},
        {(int) exploreState.varValues[8]
            + (int) exploreState.varValues[9] / Math.pow(10, decPlaces),
            (int) exploreState.varValues[10]
                + (int) exploreState.varValues[11] / Math.pow(10, decPlaces)}});


    RealMatrix P_recons = UPostPrism.multiply(DPostPrism).multiply(UPostPrism.transpose());

    // System.out.println(P_recons);
    double innov =
        (int) state.varValues[24] + ((int) state.varValues[25] / Math.pow(10, decPlaces));
    double s_inv =
        ((int) state.varValues[26] + ((int) state.varValues[27] / Math.pow(10, decPlaces)));
    //
    // // System.out.println("innov is " + innov);
    double s = ((int) state.varValues[28] + ((int) state.varValues[29] / Math.pow(10, decPlaces)));
    // //System.out.println("s is" + s);
    // // System.out.println("innov is " + innov + " s inv is " + s_inv + " s is "+
    // s);
    //
    double p_11 = ((int) state.varValues[4] + ((int) state.varValues[5] / Math.pow(10, decPlaces)));
    boolean inrange = (innov >= (-2 * Math.sqrt(s)) && (innov <= (2 * Math.sqrt(s))));
    double nis = (innov * innov * s_inv);
    // //System.out.println("nis is "+ nis);
    // if (r == 0 && (int) state.varValues[0] >= 0) {
    // return nis;
    // }
    //
    // if (r == 1 && (int) state.varValues[0] >= 0 && inrange) {
    // return 1;
    // }
    //
    // if (r == 2 && (int) state.varValues[0] >= 0) {
    // return nis / max_time;
    // }

    // double cond_number = ( (int)state.varValues[30]+ ((int)state.varValues[31] /
    // Math.pow(10, decPlaces)));

    SingularValueDecomposition svd = new SingularValueDecomposition(P_recons);
    double cond_number_prism = svd.getConditionNumber();
    boolean cond_bound_float = Math.log10(cond_number_prism) > 6;
    double dp_error_est = 2 * Precision.EPSILON * cond_number_prism;
    double inv_cond_number = svd.getInverseConditionNumber();
    // double cond_bound_sp 1.1920929e-07
    if (r == 0 && (int) state.varValues[0] >= 0) {
      System.out.println("Matrix to be checked is " + P_recons);
      // System.out.println("In reward matrix is " + UPostPrism);
      // return Math.log10(svd.getConditionNumber());
      return cond_number_prism;
    }
    if (r == 1 && (int) state.varValues[0] >= 0) {

      return inv_cond_number;
    }
    if (r == 2 && (int) state.varValues[0] >= 0) {
      return nis;
    }
    if (r == 3 && (int) state.varValues[0] >= 0 && inrange) {
      return 1;
    }

    if (r == 4 && (int) state.varValues[0] >= 0) {
      return nis / max_time;
    }
    if (r == 5 && (int) state.varValues[0] >= 0 && cond_bound_float) {
      return 1;
    }

    if (r == 6 && (int) state.varValues[0] >= 0) {
      return dp_error_est;
    }
    // for testing
    if (r == 7 && (int) state.varValues[0] >= 0)
      return 26;

    return 0;
  }

  // Not really used methods
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
