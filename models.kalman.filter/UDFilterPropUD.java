/**
 * Copyright 2001-2019 The Apache Software Foundation
 * 
 * Modifications copyright (C) 2018-2020 Alexandros Evangelidis.
 * 
 * VerFilter
 * 
 * The structure of this class is based upon the KalmanFilter class licensed by the ASF. In this
 * class the predict and correct methods have been rewritten to implement the U-D filter.
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

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixDimensionMismatchException;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.NonSquareMatrixException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.util.MathUtils;


/**
 * U-D filter algorithm is based on the based on the following book: Grewal, M.S., Andrews,
 * A.P.: Kalman Filtering: Theory and Practice with MATLAB. Wiley-IEEE Press, 4th edn. (2014).   
*/

public class UDFilterPropUD implements KalmanFilter {
  /** The process model used by this filter instance. */
  private DefaultProcessModel processModel;
  /** The measurement model used by this filter instance. */
  private final MeasurementModel measurementModel;
  /** The transition matrix, equivalent to A. */
  private RealMatrix transitionMatrix;
  /** The transposed transition matrix. */
  private RealMatrix transitionMatrixT;
  /** The control matrix, equivalent to B. */
  private RealMatrix controlMatrix;
  /** The measurement matrix, equivalent to H. */
  private RealMatrix measurementMatrix;
  /** The transposed measurement matrix. */
  private RealMatrix measurementMatrixT;
  /** The internal state estimation vector, equivalent to x hat. */
  private RealVector stateEstimation;
  /** The error covariance matrix, equivalent to P. */
  private RealMatrix U_post, D_post;
  private RealMatrix G_in;
  RealVector innovation;
  RealMatrix s;
  RealMatrix s_inv;
  static double nis;
  RealVector sv_dot;
  private double sigma2;
  private RealMatrix D_prior, U_prior;


  /**
   * Creates a new Kalman filter with the given process and measurement models.
   *
   * @param pm the model defining the underlying process dynamics
   * @param measurement the model defining the given measurement characteristics
   * @throws NullArgumentException if any of the given inputs is null (except for the control
   *         matrix)
   * @throws NonSquareMatrixException if the transition matrix is non square
   * @throws DimensionMismatchException if the column dimension of the transition matrix does not
   *         match the dimension of the initial state estimation vector
   * @throws MatrixDimensionMismatchException if the matrix dimensions do not fit together
   */
  public UDFilterPropUD(DefaultProcessModel pm, final MeasurementModel measurement, RealMatrix G_in,
      double sigma2) throws NullArgumentException, NonSquareMatrixException,
      DimensionMismatchException, MatrixDimensionMismatchException {

    MathUtils.checkNotNull(pm);
    MathUtils.checkNotNull(measurement);

    this.processModel = pm;
    this.measurementModel = measurement;
    this.G_in = G_in;
    this.sigma2 = sigma2;

    transitionMatrix = processModel.getStateTransitionMatrix();
    MathUtils.checkNotNull(transitionMatrix);
    transitionMatrixT = transitionMatrix.transpose();

    // create an empty matrix if no control matrix was given
    if (processModel.getControlMatrix() == null) {
      controlMatrix = new Array2DRowRealMatrix();
    } else {
      controlMatrix = processModel.getControlMatrix();
    }

    measurementMatrix = measurementModel.getMeasurementMatrix();
    MathUtils.checkNotNull(measurementMatrix);

    measurementMatrixT = measurementMatrix.transpose();

    // check that the process and measurement noise matrices are not null
    // they will be directly accessed from the model as they may change
    // over time
    RealMatrix processNoise = processModel.getProcessNoise();
    MathUtils.checkNotNull(processNoise);
    RealMatrix measNoise = measurementModel.getMeasurementNoise();
    MathUtils.checkNotNull(measNoise);

    // set the initial state estimate to a zero vector if it is not
    // available from the process model
    if (processModel.getInitialStateEstimate() == null) {
      stateEstimation = new ArrayRealVector(transitionMatrix.getColumnDimension());
    } else {
      stateEstimation = processModel.getInitialStateEstimate();
    }

    if (transitionMatrix.getColumnDimension() != stateEstimation.getDimension()) {
      throw new DimensionMismatchException(transitionMatrix.getColumnDimension(),
          stateEstimation.getDimension());
    }

    // initialize the error covariance to the process noise if it is not
    // available from the process model
    U_post = processModel.getU();
    D_post = processModel.getD();

    // sanity checks, the control matrix B may be null

    // A must be a square matrix
    if (!transitionMatrix.isSquare()) {
      throw new NonSquareMatrixException(transitionMatrix.getRowDimension(),
          transitionMatrix.getColumnDimension());
    }

    // row dimension of B must be equal to A
    // if no control matrix is available, the row and column dimension will be 0
    if (controlMatrix != null && controlMatrix.getRowDimension() > 0
        && controlMatrix.getColumnDimension() > 0
        && controlMatrix.getRowDimension() != transitionMatrix.getRowDimension()) {
      throw new MatrixDimensionMismatchException(controlMatrix.getRowDimension(),
          controlMatrix.getColumnDimension(), transitionMatrix.getRowDimension(),
          controlMatrix.getColumnDimension());
    }
    // Q must be equal to A
    MatrixUtils.checkAdditionCompatible(transitionMatrix,
        G_in.scalarMultiply(sigma2).multiply(G_in.transpose()));

    // column dimension of H must be equal to row dimension of A
    if (measurementMatrix.getColumnDimension() != transitionMatrix.getRowDimension()) {
      throw new MatrixDimensionMismatchException(measurementMatrix.getRowDimension(),
          measurementMatrix.getColumnDimension(), measurementMatrix.getRowDimension(),
          transitionMatrix.getRowDimension());
    }
    // row dimension of R must be equal to row dimension of H
    if (measNoise.getRowDimension() != measurementMatrix.getRowDimension()) {
      throw new MatrixDimensionMismatchException(measNoise.getRowDimension(),
          measNoise.getColumnDimension(), measurementMatrix.getRowDimension(),
          measNoise.getColumnDimension());
    }
  }

  /**
   * Returns the dimension of the state estimation vector.
   *
   * @return the state dimension
   */
  @Override
  public int getStateDimension() {
    return stateEstimation.getDimension();
  }

  /**
   * Returns the dimension of the measurement vector.
   *
   * @return the measurement vector dimension
   */
  @Override
  public int getMeasurementDimension() {
    return measurementMatrix.getRowDimension();
  }

  /**
   * Returns the current state estimation vector.
   * 
   * @return the state estimation vector
   */
  @Override
  public double[] getStateEstimation() {
    return stateEstimation.toArray();
  }

  /**
   * Returns a copy of the current state estimation vector.
   *
   * @return the state estimation vector
   */
  @Override
  public RealVector getStateEstimationVector() {
    return stateEstimation.copy();
  }

  /**
   * Returns the current error covariance matrix.
   *
   * @return the error covariance matrix
   */
  @Override
  public double[][] getErrorCovariance() {
    return null;
  }

  /**
   * Returns a copy of the current error covariance matrix.
   *
   * @return the error covariance matrix
   */
  @Override
  public RealMatrix getErrorCovarianceMatrix() {
    return null;
  }

  /**
   * Predict the internal state estimation one time step ahead.
   */
  @Override
  public void predict() {
    predict((RealVector) null);
  }

  /**
   * Predict the internal state estimation one time step ahead.
   *
   * @param u the control vector
   * @throws DimensionMismatchException if the dimension of the control vector does not fit
   */
  @Override
  public void predict(double[] u) throws DimensionMismatchException {
    predict(new ArrayRealVector(u, false));
  }

  /**
   * Predict the internal state estimation one time step ahead.
   *
   * @param u the control vector
   * @throws DimensionMismatchException if the dimension of the control vector does not match
   */
  @Override
  public void predict(RealVector u) throws DimensionMismatchException {
    // sanity checks
    if (u != null && u.getDimension() != controlMatrix.getColumnDimension()) {
      throw new DimensionMismatchException(u.getDimension(), controlMatrix.getColumnDimension());
    }
    int nd = 2;

    D_prior = MatrixUtils.createRealMatrix(2, 2);
    U_prior = MatrixUtils.createRealIdentityMatrix(2);

    stateEstimation = transitionMatrix.operate(stateEstimation);
    int n = G_in.getRowDimension();
    int r = G_in.getColumnDimension();
    RealMatrix G = G_in.copy();
    RealMatrix PhiU = transitionMatrix.multiply(U_post);
    for (int i = n - 1; i > -1; i--) {
      double sigma = 0.0;
      for (int j = 0; j < n; j++) {
        sigma = sigma + Math.pow(PhiU.getEntry(i, j), 2) * D_post.getEntry(j, j);
        if (j < r) {
          sigma = sigma + Math.pow(G.getEntry(i, j), 2) * sigma2;
        }
      }

      D_prior.setEntry(i, i, sigma);

      for (int j = 0; j < i; j++) {
        sigma = 0;
        for (int k = 0; k < n; k++) {
          sigma = sigma + PhiU.getEntry(i, k) * D_post.getEntry(k, k) * PhiU.getEntry(j, k);
        }
        for (int k = 0; k < r; k++) {
          sigma = sigma + G.getEntry(i, k) * sigma2 * G.getEntry(j, k);
        }
        U_prior.setEntry(j, i, sigma / D_prior.getEntry(i, i));

        for (int k = 0; k < n; k++) {
          PhiU.setEntry(j, k, PhiU.getEntry(j, k) - U_prior.getEntry(j, i) * PhiU.getEntry(i, k));
        }
        for (int k = 0; k < r; k++) {
          G.setEntry(j, k, G.getEntry(j, k) - U_prior.getEntry(j, i) * G.getEntry(i, k));
        }
      }
    }

  }

  /**
   * Correct the current state estimate with an actual measurement.
   *
   * @param z the measurement vector
   * @throws NullArgumentException if the measurement vector is {@code null}
   * @throws DimensionMismatchException if the dimension of the measurement vector does not fit
   * @throws SingularMatrixException if the covariance matrix could not be inverted
   */
  @Override
  public void correct(double[] z)
      throws NullArgumentException, DimensionMismatchException, SingularMatrixException {
    correct(new ArrayRealVector(z, false));
  }

  /**
   * Correct the current state estimate with an actual measurement.
   *
   * @param z the measurement vector
   * @throws NullArgumentException if the measurement vector is {@code null}
   * @throws DimensionMismatchException if the dimension of the measurement vector does not fit
   * @throws SingularMatrixException if the covariance matrix could not be inverted
   */
  @Override
  public void correct(RealVector z)
      throws NullArgumentException, DimensionMismatchException, SingularMatrixException {
    // sanity checks
    MathUtils.checkNotNull(z);
    if (z.getDimension() != measurementMatrix.getRowDimension()) {
      throw new DimensionMismatchException(z.getDimension(), measurementMatrix.getRowDimension());
    }

    RealMatrix a = U_prior.transpose().multiply(measurementMatrix.transpose());
    RealMatrix b = D_prior.multiply(a);

    U_post = U_prior.copy();
    D_post = D_prior.copy();
    innovation = z.subtract(measurementMatrix.operate(stateEstimation));
    double alpha = measurementModel.getMeasurementNoise().getEntry(0, 0);
    double gamma = 1 / alpha;

    for (int i = 0; i < stateEstimation.getDimension(); i++) {

      double beta = alpha;
      alpha = alpha + a.getEntry(i, 0) * b.getEntry(i, 0);
      double lambda = -(a.getEntry(i, 0)) * gamma;
      gamma = 1 / alpha;
      D_post.setEntry(i, i, beta * gamma * D_post.getEntry(i, i));

      for (int j = 0; j < i; j++) {
        beta = U_post.getEntry(j, i);

        U_post.setEntry(j, i, beta + b.getEntry(j, 0) * lambda);
        b.setEntry(j, j, b.getEntry(j, 0) + b.getEntry(i, 0) * beta);
      }
    }

    stateEstimation = stateEstimation.add(b.operate(innovation.mapMultiply(gamma)));

    RealMatrix errorCovariance = U_post.multiply(D_post).multiply(U_post.transpose());
  }

  @Override
  public RealVector getInnovation() {

    return innovation;
  }

  public RealMatrix getU() {
    return U_post;

  }

  public RealMatrix getD() {
    return D_post;
  }

  @Override
  public RealMatrix getInnovationCovariance() {
    return s;
  }

  @Override
  public RealMatrix getInnovationCovarianceInv() {
    return s_inv;
  }

  @Override
  public void setProcessModel(DefaultProcessModel process) {
    this.processModel = process;
  }

  public static double getNIS() {
    return nis;
  }
}
