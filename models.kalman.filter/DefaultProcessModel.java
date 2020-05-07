/**
 * Copyright 2001-2019 The Apache Software Foundation
 * 
 * Modifications copyright (C) 2019 Alexandros Evangelidis.
 * 
 * VerFilter
 * 
 * This file has been modified by adding the getC() method.
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
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.filter.ProcessModel;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import demos.Computations;

/**
 * Default implementation of a {@link ProcessModel} for the use with a {@link KalmanFilter}.
 *
 * @since 3.0
 *
 *        This class has been modified/extended by Alexandros Evangelidis.
 */
public class DefaultProcessModel implements ProcessModel {
  /**
   * The state transition matrix, used to advance the internal state estimation each time-step.
   */
  private RealMatrix stateTransitionMatrix;

  /**
   * The control matrix, used to integrate a control input into the state estimation.
   */
  private RealMatrix controlMatrix;

  /** The process noise covariance matrix. */
  private RealMatrix processNoiseCovMatrix;

  /** The initial state estimation of the observed process. */
  private RealVector initialStateEstimateVector;

  /** The initial error covariance matrix of the observed process. */
  private RealMatrix initialErrorCovMatrix;

  /**
   * Create a new {@link ProcessModel}, taking double arrays as input parameters.
   *
   * @param stateTransition the state transition matrix
   * @param control the control matrix
   * @param processNoise the process noise matrix
   * @param initialStateEstimate the initial state estimate vector
   * @param initialErrorCovariance the initial error covariance matrix
   * @throws NullArgumentException if any of the input arrays is {@code null}
   * @throws NoDataException if any row / column dimension of the input matrices is zero
   * @throws DimensionMismatchException if any of the input matrices is non-rectangular
   */
  public DefaultProcessModel(final double[][] stateTransition, final double[][] control,
      final double[][] processNoise, final double[] initialStateEstimate,
      final double[][] initialErrorCovariance)
      throws NullArgumentException, NoDataException, DimensionMismatchException {

    this(new Array2DRowRealMatrix(stateTransition), new Array2DRowRealMatrix(control),
        new Array2DRowRealMatrix(processNoise), new ArrayRealVector(initialStateEstimate),
        new Array2DRowRealMatrix(initialErrorCovariance));
  }

  /**
   * Create a new {@link ProcessModel}, taking double arrays as input parameters.
   * <p>
   * The initial state estimate and error covariance are omitted and will be initialized by the
   * {@link KalmanFilter} to default values.
   *
   * @param stateTransition the state transition matrix
   * @param control the control matrix
   * @param processNoise the process noise matrix
   * @throws NullArgumentException if any of the input arrays is {@code null}
   * @throws NoDataException if any row / column dimension of the input matrices is zero
   * @throws DimensionMismatchException if any of the input matrices is non-rectangular
   */
  public DefaultProcessModel(final double[][] stateTransition, final double[][] control,
      final double[][] processNoise)
      throws NullArgumentException, NoDataException, DimensionMismatchException {

    this(new Array2DRowRealMatrix(stateTransition), new Array2DRowRealMatrix(control),
        new Array2DRowRealMatrix(processNoise), null, null);
  }

  /**
   * Create a new {@link ProcessModel}, taking double arrays as input parameters.
   *
   * @param stateTransition the state transition matrix
   * @param control the control matrix
   * @param processNoise the process noise matrix
   * @param initialStateEstimate the initial state estimate vector
   * @param initialErrorCovariance the initial error covariance matrix
   */
  public DefaultProcessModel(RealMatrix stateTransition, RealMatrix control,
      RealMatrix processNoise, RealVector initialStateEstimate, RealMatrix initialErrorCovariance) {
    this.stateTransitionMatrix = stateTransition;
    this.controlMatrix = control;
    this.processNoiseCovMatrix = processNoise;
    this.initialStateEstimateVector = initialStateEstimate;
    this.initialErrorCovMatrix = initialErrorCovariance;
  }

  public RealMatrix getProcessNoiseCovMatrix() {
    return processNoiseCovMatrix;
  }

  public void setProcessNoiseCovMatrix(RealMatrix processNoiseCovMatrix) {
    this.processNoiseCovMatrix = processNoiseCovMatrix;
  }

  public RealVector getInitialStateEstimateVector() {
    return initialStateEstimateVector;
  }

  public void setInitialStateEstimateVector(RealVector initialStateEstimateVector) {
    this.initialStateEstimateVector = initialStateEstimateVector;
  }

  public RealMatrix getInitialErrorCovMatrix() {
    return initialErrorCovMatrix;
  }

  public void setInitialErrorCovMatrix(RealMatrix initialErrorCovMatrix) {
    this.initialErrorCovMatrix = initialErrorCovMatrix;
  }

  public void setStateTransitionMatrix(RealMatrix stateTransitionMatrix) {
    this.stateTransitionMatrix = stateTransitionMatrix;
  }

  public void setControlMatrix(RealMatrix controlMatrix) {
    this.controlMatrix = controlMatrix;
  }

  public RealMatrix getC() {
    return Computations.getUTCholeskyFactor(initialErrorCovMatrix);
  }

  /** {@inheritDoc} */
  @Override
  public RealMatrix getStateTransitionMatrix() {
    return stateTransitionMatrix;
  }

  /** {@inheritDoc} */
  @Override
  public RealMatrix getControlMatrix() {
    return controlMatrix;
  }

  /** {@inheritDoc} */
  @Override
  public RealMatrix getProcessNoise() {
    return processNoiseCovMatrix;
  }

  /** {@inheritDoc} */
  @Override
  public RealVector getInitialStateEstimate() {
    return initialStateEstimateVector;
  }

  /** {@inheritDoc} */
  @Override
  public RealMatrix getInitialErrorCovariance() {
    return initialErrorCovMatrix;
  }
}
