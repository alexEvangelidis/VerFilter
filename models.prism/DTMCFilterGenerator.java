/**
 * Copyright (C) Alexandros Evangelidis.
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

package models.prism;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.filter.DefaultMeasurementModel;
import org.apache.commons.math3.filter.MeasurementModel;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import models.kalman.filter.DefaultProcessModel;
import prism.Prism;
import prism.PrismDevNullLog;
import prism.PrismException;
import prism.PrismLog;

/**
 * This class runs the experiments presented in the FM paper.
 *
 */
public class DTMCFilterGenerator {

  public static void main(String[] args) throws IllegalArgumentException, IllegalStateException,
      InterruptedException, ExecutionException, IOException {
    new DTMCFilterGenerator().run();
  }

  public void run() throws IllegalArgumentException, IllegalStateException, InterruptedException,
      ExecutionException, IOException {
    try {
      double dt = 1;

      RealMatrix F = new Array2DRowRealMatrix(new double[][] {{1, dt}, {0, 1}});
      RealVector x = new ArrayRealVector(new double[] {5, 5});
      RealMatrix B = null;

      double sigma2 = 0.001;

      RealMatrix Q = new Array2DRowRealMatrix(new double[][] {{0.00025, 0.0005}, {0.0005, 0.001}});
      RealMatrix P = new Array2DRowRealMatrix(new double[][] {{10, 0}, {0, 10}});
      RealMatrix H = new Array2DRowRealMatrix(new double[][] {{1d, 0d}});
      RealMatrix R = new Array2DRowRealMatrix(new double[] {0.001});

      MeasurementModel mm = new DefaultMeasurementModel(H, R);


      DefaultProcessModel pm = new DefaultProcessModel(F, B, Q, x, P);

      // Process noise distribution
      NormalDistribution nd_proc_noise = new NormalDistribution(0, Math.sqrt(sigma2));
      // Measurement noise distribution
      NormalDistribution nd_meas_noise = new NormalDistribution(0, Math.sqrt(R.getEntry(0, 0)));


      RealMatrix Gamma = new Array2DRowRealMatrix(new double[][] {{Math.pow(dt, 2) / 2.}, {dt}});
      SchmidtCarlsonFilterPrismPropC modelGen = new SchmidtCarlsonFilterPrismPropC(2, 3, sigma2, dt,
          Gamma, nd_proc_noise, nd_meas_noise, pm, mm, 3);


      PrismLog mainLog = new PrismDevNullLog();
      // Initialise PRISM engine
      Prism prism = new Prism(mainLog);
      prism.loadModelGenerator(modelGen);
      prism.setEngine(Prism.EXPLICIT);
      prism.initialise();

      // export the model to a dot file (which triggers its construction)
      prism.exportTransToFile(true, Prism.EXPORT_DOT_STATES, new File("dtmc.dot"));
      prism.exportStatesToFile(0, new File("states.txt"));
      prism.exportStateRewardsToFile(0, new File("staterewards.txt"));
      String[] props = new String[] {"R{\"cond\"}=? [I=3]", "P=?[G \"isPD\"]"};

      for (String prop : props) {
        System.out.println(prop + ":");
        System.out.println((double) prism.modelCheck(prop).getResult());
      }
      prism.closeDown();

    } catch (FileNotFoundException e) {
      System.out.println("Error: " + e.getMessage());
      System.exit(1);
    } catch (PrismException e) {
      System.out.println("Error: " + e.getMessage());
      System.exit(1);
    }
  }

}
