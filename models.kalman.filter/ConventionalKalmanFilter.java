/**
 * Copyright 2001-2019 The Apache Software Foundation
 * 
 * Modifications copyright (C) 2019 Alexandros Evangelidis.
 * 
 * VerFilter
 * 
 * This file is based on the original KalmanFilter class. The correct method has been modified in
 * this class.
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

package models.kalman.filter;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.filter.MeasurementModel;
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
 *
 * This class implements a Kalman filter.
 *
 */
public class ConventionalKalmanFilter implements models.kalman.filter.KalmanFilter {
  /** The process model used by this filter instance. */
  private DefaultProcessModel processModel;
  /** The measurement model used by this filter instance. */
  private final MeasurementModel measurementModel;
  /** The transition matrix, equivalent to A. */
  private RealMatrix transitionMatrix;
  /** The control matrix, equivalent to B. */
  private RealMatrix controlMatrix;
  /** The measurement matrix, equivalent to H. */
  private RealMatrix measurementMatrix;
  /** The transposed measurement matrix. */
  private RealMatrix measurementMatrixT;
  /** The internal state estimation vector, equivalent to x hat. */
  private RealVector stateEstimation;
  /** The error covariance matrix, equivalent to P. */
  private RealMatrix errorCovariance;

  private RealVector innovation;
  private RealMatrix s;
  private RealMatrix s_inv;

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
  public ConventionalKalmanFilter(DefaultProcessModel pm, final MeasurementModel measurement)
      throws NullArgumentException, NonSquareMatrixException, DimensionMismatchException,
      MatrixDimensionMismatchException {

    MathUtils.checkNotNull(pm);
    MathUtils.checkNotNull(measurement);

    this.processModel = pm;
    this.measurementModel = measurement;

    transitionMatrix = processModel.getStateTransitionMatrix();
    MathUtils.checkNotNull(transitionMatrix);
    transitionMatrix.transpose();

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
    if (processModel.getInitialErrorCovariance() == null) {
      errorCovariance = processNoise.copy();
    } else {
      errorCovariance = processModel.getInitialErrorCovariance();
    }

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
    MatrixUtils.checkAdditionCompatible(transitionMatrix, processNoise);

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
   *
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
    return errorCovariance.getData();
  }

  /**
   * Returns a copy of the current error covariance matrix.
   *
   * @return the error covariance matrix
   */
  @Override
  public RealMatrix getErrorCovarianceMatrix() {
    return errorCovariance.copy();
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
    // project the state estimation ahead (a priori state)
    // x(k)^- = F * x(k-1)^+
    stateEstimation = transitionMatrix.operate(stateEstimation);
    // add control input if it is available
    if (u != null) {
      stateEstimation = stateEstimation.add(controlMatrix.operate(u));
    }
    // project the error covariance ahead
    // P(k)^- = F * P(k-1)^+ * F' + Q
    errorCovariance = transitionMatrix.multiply(errorCovariance)
        .multiply(transitionMatrix.transpose()).add(processModel.getProcessNoise());
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

    // S = H * P(k)^- * H' + R
    s = measurementMatrix.multiply(errorCovariance).multiply(measurementMatrixT)
        .add(measurementModel.getMeasurementNoise());

    // y = z(k) - H * x(k)-
    innovation = z.subtract(measurementMatrix.operate(stateEstimation));

    // calculate gain matrix
    // K(k) = P(k)^- * H' * (H * P(k)^- * H' + R)^-1
    // K(k) = P(k)^- * H' * S^-1
    RealMatrix kalmanGain =
        errorCovariance.multiply(measurementMatrix.transpose()).multiply(MatrixUtils.inverse(s));

    // xHat(k) = xHat(k)- + K * Innovation
    // x(k)^+ = x(k_^- + K * y
    stateEstimation = stateEstimation.add(kalmanGain.operate(innovation));

    // P(k)^+ = (I - K(k) * H) * P(k)^-
    RealMatrix identity = MatrixUtils.createRealIdentityMatrix(kalmanGain.getRowDimension());
    errorCovariance =
        identity.subtract(kalmanGain.multiply(measurementMatrix)).multiply(errorCovariance);
  }

  @Override
  public RealVector getInnovation() {
    return innovation;
  }

  @Override
  public RealMatrix getInnovationCovariance() {
    return s;
  }

  @Override
  public void setProcessModel(DefaultProcessModel process) {
    this.processModel = process;
  }

  @Override
  public RealMatrix getInnovationCovarianceInv() {
    return s_inv;
  }
}
