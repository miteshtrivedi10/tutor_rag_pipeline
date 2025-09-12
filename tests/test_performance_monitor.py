import unittest
import time
from src.rag.performance_monitor import PerformanceMonitor, PerformanceTimer


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for PerformanceMonitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(max_metrics_history=10)
    
    def test_record_operation_success(self):
        """Test recording a successful operation."""
        self.monitor.record_operation("test_operation", 0.5, True, None, 100, 200)
        
        # Verify metrics were recorded
        self.assertEqual(len(self.monitor.metrics_history), 1)
        metric = self.monitor.metrics_history[0]
        self.assertEqual(metric.operation_name, "test_operation")
        self.assertEqual(metric.execution_time, 0.5)
        self.assertTrue(metric.success)
        self.assertIsNone(metric.error_message)
        self.assertEqual(metric.input_size, 100)
        self.assertEqual(metric.output_size, 200)
    
    def test_record_operation_failure(self):
        """Test recording a failed operation."""
        self.monitor.record_operation("test_operation", 0.5, False, "Test error", 100, 200)
        
        # Verify metrics were recorded
        self.assertEqual(len(self.monitor.metrics_history), 1)
        metric = self.monitor.metrics_history[0]
        self.assertEqual(metric.operation_name, "test_operation")
        self.assertEqual(metric.execution_time, 0.5)
        self.assertFalse(metric.success)
        self.assertEqual(metric.error_message, "Test error")
        self.assertEqual(metric.input_size, 100)
        self.assertEqual(metric.output_size, 200)
    
    def test_get_average_time(self):
        """Test getting average execution time."""
        # Record multiple operations
        self.monitor.record_operation("test_operation", 1.0, True)
        self.monitor.record_operation("test_operation", 2.0, True)
        self.monitor.record_operation("test_operation", 3.0, True)
        
        # Verify average time
        avg_time = self.monitor.get_average_time("test_operation")
        self.assertEqual(avg_time, 2.0)
    
    def test_get_average_time_no_data(self):
        """Test getting average execution time with no data."""
        avg_time = self.monitor.get_average_time("nonexistent_operation")
        self.assertIsNone(avg_time)
    
    def test_get_success_rate(self):
        """Test getting success rate."""
        # Record successful and failed operations
        self.monitor.record_operation("test_operation", 1.0, True)
        self.monitor.record_operation("test_operation", 2.0, False, "Error")
        self.monitor.record_operation("test_operation", 3.0, True)
        
        # Verify success rate
        success_rate = self.monitor.get_success_rate("test_operation")
        self.assertEqual(success_rate, 66.66666666666666)
    
    def test_get_success_rate_no_data(self):
        """Test getting success rate with no data."""
        success_rate = self.monitor.get_success_rate("nonexistent_operation")
        self.assertEqual(success_rate, 100.0)
    
    def test_get_recent_metrics(self):
        """Test getting recent metrics."""
        # Record multiple operations
        for i in range(5):
            self.monitor.record_operation(f"operation_{i}", float(i), True)
        
        # Get recent metrics
        recent_metrics = self.monitor.get_recent_metrics(3)
        self.assertEqual(len(recent_metrics), 3)
        self.assertEqual(recent_metrics[0].operation_name, "operation_2")
        self.assertEqual(recent_metrics[1].operation_name, "operation_3")
        self.assertEqual(recent_metrics[2].operation_name, "operation_4")
    
    def test_get_operation_summary(self):
        """Test getting operation summary."""
        # Record multiple operations
        self.monitor.record_operation("test_operation", 1.0, True)
        self.monitor.record_operation("test_operation", 2.0, True)
        self.monitor.record_operation("test_operation", 3.0, False, "Error")
        
        # Get summary
        summary = self.monitor.get_operation_summary()
        self.assertIn("test_operation", summary)
        operation_summary = summary["test_operation"]
        self.assertEqual(operation_summary["average_time"], 1.5)  # Only successful operations
        self.assertEqual(operation_summary["min_time"], 1.0)
        self.assertEqual(operation_summary["max_time"], 2.0)
        self.assertEqual(operation_summary["total_executions"], 2)  # Only successful operations
        self.assertEqual(operation_summary["success_rate"], 66.66666666666666)
    
    def test_clear_history(self):
        """Test clearing history."""
        # Record some operations
        self.monitor.record_operation("test_operation", 1.0, True)
        self.monitor.record_operation("test_operation", 2.0, True)
        
        # Clear history
        self.monitor.clear_history()
        
        # Verify history is cleared
        self.assertEqual(len(self.monitor.metrics_history), 0)
        self.assertEqual(len(self.monitor.operation_stats), 0)
    
    def test_history_limit(self):
        """Test that history is limited to max_metrics_history."""
        # Record more operations than the limit
        for i in range(15):
            self.monitor.record_operation("test_operation", float(i), True)
        
        # Verify only the most recent metrics are kept
        self.assertEqual(len(self.monitor.metrics_history), 10)
        self.assertEqual(self.monitor.metrics_history[0].execution_time, 5.0)
        self.assertEqual(self.monitor.metrics_history[-1].execution_time, 14.0)


class TestPerformanceTimer(unittest.TestCase):
    """Test cases for PerformanceTimer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
    
    def test_timer_success(self):
        """Test timer with successful operation."""
        with PerformanceTimer(self.monitor, "test_operation", 100) as timer:
            time.sleep(0.01)  # Sleep for 10ms
            timer.set_output_size(200)
        
        # Verify metrics were recorded
        self.assertEqual(len(self.monitor.metrics_history), 1)
        metric = self.monitor.metrics_history[0]
        self.assertEqual(metric.operation_name, "test_operation")
        self.assertTrue(metric.success)
        self.assertIsNone(metric.error_message)
        self.assertEqual(metric.input_size, 100)
        self.assertEqual(metric.output_size, 200)
        self.assertGreater(metric.execution_time, 0.01)
    
    def test_timer_failure(self):
        """Test timer with failed operation."""
        try:
            with PerformanceTimer(self.monitor, "test_operation", 100) as timer:
                timer.set_output_size(200)
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected exception
        
        # Verify metrics were recorded
        self.assertEqual(len(self.monitor.metrics_history), 1)
        metric = self.monitor.metrics_history[0]
        self.assertEqual(metric.operation_name, "test_operation")
        self.assertFalse(metric.success)
        self.assertEqual(metric.error_message, "Test error")
        self.assertEqual(metric.input_size, 100)
        self.assertEqual(metric.output_size, 200)


if __name__ == '__main__':
    unittest.main()