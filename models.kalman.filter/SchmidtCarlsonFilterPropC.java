/**
Copyright 2001-2019 The Apache Software Foundation

 Modifications copyright (C) 2019 Alexandros Evangelidis.

 VerFilter

 The structure of this class is based upon the KalmanFilter class
 licensed by the ASF. In this class the predict and correct methods
 have been rewritten to implement the Carlson-Schmidt square-root filter.

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
 * Schmidt-Carlson algorithm is based on the based on the following book:
 * Grewal, M.S., Andrews, A.P.: Kalman Filtering: Theory and Practice with
 * MATLAB. Wiley-IEEE Press, 4th edn. (2014). Also, see Carlson's original paper
 * Maybeck, P.S.: Stochastic Carlson, N.A.: Fast triangular formulation of the
 * square root filter. AIAA J. 11(9), 1259–1265 (1973).
 *
 */
public class SchmidtCarlsonFilterPropC implements KalmanFilter {
	private DefaultProcessModel processModel;
	/** The measurement model used by this filter instance. */
	private final MeasurementModel measurementModel;
	/** The transition matrix, equivalent to A. */
	private RealMatrix transitionMatrix;
	/** The control matrix, equivalent to B. */
	private RealMatrix controlMatrix;
	/** The measurement matrix, equivalent to H. */
	private RealMatrix measurementMatrix;
	/** The internal state estimation vector, equivalent to x hat. */
	private RealVector stateEstimation;
	/** The error covariance matrix, equivalent to P. */
	private RealMatrix C;
	private RealMatrix G_in;
	private double sigma2;
	private RealVector innovation;
	private RealMatrix s;
	private RealMatrix s_inv;

	/**
	 * Creates a new Kalman filter with the given process and measurement models.
	 *
	 * @param pm          the model defining the underlying process dynamics
	 * @param measurement the model defining the given measurement characteristics
	 * @throws NullArgumentException            if any of the given inputs is null
	 *                                          (except for the control matrix)
	 * @throws NonSquareMatrixException         if the transition matrix is non
	 *                                          square
	 * @throws DimensionMismatchException       if the column dimension of the
	 *                                          transition matrix does not match the
	 *                                          dimension of the initial state
	 *                                          estimation vector
	 * @throws MatrixDimensionMismatchException if the matrix dimensions do not fit
	 *                                          together
	 */
	public SchmidtCarlsonFilterPropC(DefaultProcessModel pm, final MeasurementModel measurement, RealMatrix G_in,
			double sigma2) throws NullArgumentException, NonSquareMatrixException, DimensionMismatchException,
			MatrixDimensionMismatchException {

		MathUtils.checkNotNull(pm);
		MathUtils.checkNotNull(measurement);

		this.processModel = pm;
		this.measurementModel = measurement;
		this.G_in = G_in;
		this.sigma2 = sigma2;

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

		measurementMatrix.transpose();

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
			throw new DimensionMismatchException(transitionMatrix.getColumnDimension(), stateEstimation.getDimension());
		}

		C = processModel.getC();

		// sanity checks, the control matrix B may be null

		// A must be a square matrix
		if (!transitionMatrix.isSquare()) {
			throw new NonSquareMatrixException(transitionMatrix.getRowDimension(),
					transitionMatrix.getColumnDimension());
		}

		// row dimension of B must be equal to A
		// if no control matrix is available, the row and column dimension will be 0
		if (controlMatrix != null && controlMatrix.getRowDimension() > 0 && controlMatrix.getColumnDimension() > 0
				&& controlMatrix.getRowDimension() != transitionMatrix.getRowDimension()) {
			throw new MatrixDimensionMismatchException(controlMatrix.getRowDimension(),
					controlMatrix.getColumnDimension(), transitionMatrix.getRowDimension(),
					controlMatrix.getColumnDimension());
		}
		// Q must be equal to A
		MatrixUtils.checkAdditionCompatible(transitionMatrix, G_in.scalarMultiply(sigma2).multiply(G_in.transpose()));

		// column dimension of H must be equal to row dimension of A
		if (measurementMatrix.getColumnDimension() != transitionMatrix.getRowDimension()) {
			throw new MatrixDimensionMismatchException(measurementMatrix.getRowDimension(),
					measurementMatrix.getColumnDimension(), measurementMatrix.getRowDimension(),
					transitionMatrix.getRowDimension());
		}
		// row dimension of R must be equal to row dimension of H
		if (measNoise.getRowDimension() != measurementMatrix.getRowDimension()) {
			throw new MatrixDimensionMismatchException(measNoise.getRowDimension(), measNoise.getColumnDimension(),
					measurementMatrix.getRowDimension(), measNoise.getColumnDimension());
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
	 * Returns a copy of the current error covariance matrix.
	 *
	 * @return the error covariance matrix
	 */

	public RealMatrix getC() {
		return C.copy();
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
	 * @throws DimensionMismatchException if the dimension of the control vector
	 *                                    does not fit
	 */
	@Override
	public void predict(double[] u) throws DimensionMismatchException {
		predict(new ArrayRealVector(u, false));
	}

	/**
	 * Predict the internal state estimation one time step ahead.
	 *
	 * @param u the control vector
	 * @throws DimensionMismatchException if the dimension of the control vector
	 *                                    does not match
	 */
	@Override
	public void predict(RealVector u) throws DimensionMismatchException {
		// sanity checks
		if (u != null && u.getDimension() != controlMatrix.getColumnDimension()) {
			throw new DimensionMismatchException(u.getDimension(), controlMatrix.getColumnDimension());
		}
		RealMatrix CPin = C;
		RealMatrix PhiCPin = transitionMatrix.multiply(CPin);

		RealMatrix GCQ = G_in.scalarMultiply(Math.sqrt(sigma2));

		int rows = GCQ.getRowDimension();
		int cols = GCQ.getColumnDimension();
		RealMatrix w = MatrixUtils.createRealMatrix(1, 2);
		RealMatrix v = MatrixUtils.createRealMatrix(1, 2);

		for (int i = rows - 1; i > -1; i--) {
			double ssigma2 = 0;

			for (int j = 0; j < cols; j++) {
				ssigma2 = ssigma2 + Math.pow(GCQ.getEntry(i, j), 2);
			}

			for (int j = 0; j <= i; j++) {
				ssigma2 = ssigma2 + Math.pow(PhiCPin.getEntry(i, j), 2);
			}

			double alpha = Math.sqrt(ssigma2);
			ssigma2 = 0;

			for (int j = 0; j < cols; j++) {
				w.setEntry(0, j, GCQ.getEntry(i, j));
				ssigma2 = ssigma2 + Math.pow(w.getEntry(0, j), 2);
			}

			for (int j = 0; j <= i; j++) {

				if (j == i) {
					v.setEntry(0, j, PhiCPin.getEntry(i, j) - alpha);
				} else {
					v.setEntry(0, j, PhiCPin.getEntry(i, j));
				}
				ssigma2 = ssigma2 + Math.pow(v.getEntry(0, j), 2);
			}
			alpha = 2. / ssigma2;

			for (int k = 0; k <= i; k++) {
				ssigma2 = 0;
				for (int j = 0; j < cols; j++) {
					ssigma2 = ssigma2 + GCQ.getEntry(k, j) * w.getEntry(0, j);
				}
				for (int j = 0; j <= i; j++) {
					ssigma2 = ssigma2 + PhiCPin.getEntry(k, j) * v.getEntry(0, j);
				}
				double beta = alpha * ssigma2;

				for (int j = 0; j < cols; j++) {
					GCQ.setEntry(k, j, GCQ.getEntry(k, j) - beta * w.getEntry(0, j));
				}
				for (int j = 0; j <= i; j++) {
					PhiCPin.setEntry(k, j, PhiCPin.getEntry(k, j) - beta * v.getEntry(0, j));
				}
			}
		}
		RealMatrix CPout = PhiCPin;
		stateEstimation = transitionMatrix.operate(stateEstimation);
		C = CPout;
	}

	/**
	 * Correct the current state estimate with an actual measurement.
	 *
	 * @param z the measurement vector
	 * @throws NullArgumentException      if the measurement vector is {@code null}
	 * @throws DimensionMismatchException if the dimension of the measurement vector
	 *                                    does not fit
	 * @throws SingularMatrixException    if the covariance matrix could not be
	 *                                    inverted
	 */
	@Override
	public void correct(double[] z) throws NullArgumentException, DimensionMismatchException, SingularMatrixException {
		correct(new ArrayRealVector(z, false));
	}

	/**
	 * Correct the current state estimate with an actual measurement.
	 *
	 * @param z the measurement vector
	 * @throws NullArgumentException      if the measurement vector is {@code null}
	 * @throws DimensionMismatchException if the dimension of the measurement vector
	 *                                    does not fit
	 * @throws SingularMatrixException    if the covariance matrix could not be
	 *                                    inverted
	 */
	@Override
	public void correct(RealVector z)
			throws NullArgumentException, DimensionMismatchException, SingularMatrixException {
		MathUtils.checkNotNull(z);
		if (z.getDimension() != measurementMatrix.getRowDimension()) {
			throw new DimensionMismatchException(z.getDimension(), measurementMatrix.getRowDimension());
		}
		double alpha = measurementModel.getMeasurementNoise().getEntry(0, 0);
		innovation = z.copy();
		RealMatrix w = MatrixUtils.createRealMatrix(1, 2);
		for (int i = 0; i < stateEstimation.getDimension(); i++) {
			innovation.setEntry(0,
					innovation.getEntry(0) - ((measurementMatrix.getEntry(0, i) * stateEstimation.getEntry(i))));

			double ssigma2 = 0;
			for (int j = 0; j <= i; j++) {
				ssigma2 = ssigma2 + C.getEntry(j, i) * measurementMatrix.getEntry(0, j);
			}
			double beta = alpha;
			alpha = alpha + Math.pow(ssigma2, 2);
			double gamma = Math.sqrt(alpha * beta);
			double eta = beta / gamma;

			double zeta = ssigma2 / gamma;

			w.setEntry(0, i, 0);

			for (int j = 0; j <= i; j++) {
				double tau = C.getEntry(j, i);
				C.setEntry(j, i, eta * C.getEntry(j, i) - zeta * w.getEntry(0, j));
				w.setEntry(0, j, w.getEntry(0, j) + tau * ssigma2);
			}
		}

		RealVector epsilon = innovation.mapDivide(alpha);
		stateEstimation = stateEstimation.add(w.transpose().operate(epsilon));
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
	public RealMatrix getInnovationCovarianceInv() {
		return s_inv;
	}

	@Override
	public void setProcessModel(DefaultProcessModel process) {
		this.processModel = process;
	}

	@Override
	public double[][] getErrorCovariance() {
		return null;
	}

	@Override
	public RealMatrix getErrorCovarianceMatrix() {
		return null;
	}
}

