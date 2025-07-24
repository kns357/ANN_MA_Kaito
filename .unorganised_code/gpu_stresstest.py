import torch
import time

def stress_test_gpu(duration_sec=300, tensor_size=(8192, 8192)):
    if not torch.cuda.is_available():
        print("CUDA not available. Aborting test.")
        return
    
    device = torch.device('cuda')
    print(f"Starting GPU stress test on {device} for {duration_sec} seconds...")
    
    a = torch.randn(tensor_size, device=device)
    b = torch.randn(tensor_size, device=device)
    
    start_time = time.time()
    iterations = 0
    try:
        while time.time() - start_time < duration_sec:
            c = torch.matmul(a, b)
            d = c * a
            s = d.sum()
            s.backward() if s.requires_grad else None
            torch.cuda.synchronize()
            iterations += 1
            if iterations % 10 == 0:
                print(f"Iteration {iterations} done")
    except Exception as e:
        print(f"Error during stress test: {e}")
        return
    
    print(f"Stress test complete: {iterations} iterations in {duration_sec} seconds")

if __name__ == "__main__":
    stress_test_gpu()
