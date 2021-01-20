import unittest
import numpy as np
import os, sys, time, resource, platform
import logging

from models import ModelPipes

test_data = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 25.0, 0.0, 1.0, 1.0, 1.0, 12.0, 5.0, 12.0, -1.0, 0.0, 24.0, 17.0])

class modelTest(unittest.TestCase):
    def test_model(self):
        logging.basicConfig(filename='tests/tests.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

        start_time = time.time()

        logger = logging.getLogger("TestModelPipelineLog")
        logger.setLevel(logging.DEBUG)

        # Create console handler and set level to Warning 
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        try:
            logger.info("Models test started.")

            models = ModelPipes().load_pipes()
            logger.info("Baseline, XGBoost, Neural Network setup finished.")

            self.assertIsNotNone(models.baseline.clf)
            logger.info("Baseline pipe checked.")
            self.assertIsNotNone(models.xgboost.clf)
            logger.info("XGBoost pipe checked.")
            self.assertIsNotNone(models.nn.clf)
            logger.info("Neural network pipe checked.")

            self.assertGreaterEqual(models.baseline.classify_insurance(test_data)["Car Insurance Probability"], 0)
            self.assertGreaterEqual(models.xgboost.classify_insurance(test_data)["Car Insurance Probability"], 0)
            self.assertGreaterEqual(models.nn.classify_insurance(test_data)["Car Insurance Probability"], 0)
            logger.info("Prediction return values checked.")

            models.eval_pipes()
            logger.info("Evaluation method checked.")

            op_system = platform.system().lower()
            max_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30) if op_system == 'darwin' else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20)
            logger.info("Maximum memory usage: {:.4f} GB.".format(max_usage))
            logger.info("Finished in {:.4f} sec.".format(time.time() - start_time))

        except Exception as e:
            logger.exception("Exception {} occured.".format(e))

if __name__ == '__main__':
    unittest.TextTestRunner().run(modelTest())  