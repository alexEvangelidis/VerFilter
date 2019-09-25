/**
Copyright 2001-2019 The Apache Software Foundation

Modifications copyright (C) 2019 Alexandros Evangelidis.

 VerFilter

 The original file is a Kalman filter class. Here, I have
 modified it by making it an interface to serve my purpose.

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

import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;

public interface KalmanFilter {
	public int getStateDimension();
	public int getMeasurementDimension();
	public double[] getStateEstimation();
	public RealVector getStateEstimationVector();
	public double[][] getErrorCovariance();
	public RealMatrix getErrorCovarianceMatrix();
	public void predict() throws DimensionMismatchException, CancellationException, InterruptedException, ExecutionException;
	public void predict(double[] u) throws DimensionMismatchException, NullArgumentException, CancellationException, InterruptedException, ExecutionException;
	public void predict(RealVector u) throws DimensionMismatchException, CancellationException, InterruptedException, ExecutionException;
	public void correct(double[] z) throws NullArgumentException, DimensionMismatchException, SingularMatrixException, CancellationException, InterruptedException, ExecutionException;
	public void correct(RealVector z) throws NullArgumentException, DimensionMismatchException, SingularMatrixException, CancellationException, InterruptedException, ExecutionException;
	public RealVector getInnovation();
	public RealMatrix getInnovationCovariance();
	public RealMatrix getInnovationCovarianceInv();
	public void setProcessModel(DefaultProcessModel process);

}
