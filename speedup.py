import concurrent.futures
"""
使用cpu多线程处理数据
USAGE:
############### BEFORE ###############
res_list = []
for data in tqdm(data_list):
    res_list.append(do_something(data))

############### AFTER ###############
from utils.speedup import concurrent_execute
@concurrent_execute(n_thread=20)
def process_data(data_list):
    res_list = []
    for data in tqdm(data_list):
        res_list.append(do_something(data))
    return res_list
process_data(data_list)
"""


def split_list(lst, n):
    """Split a list into n roughly equal parts"""
    avg = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    for i in range(n):
        end = start + avg + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result


def merge_results(results):
    """Merge results based on their type"""
    if isinstance(results[0], dict):
        # Merge dictionaries
        merged_result = {}
        for result in results:
            for key, value in result.items():
                if key not in merged_result:
                    merged_result[key] = value
                else:
                    if isinstance(value, list):
                        merged_result[key].extend(value)
                    elif isinstance(value, dict):
                        merged_result[key].update(value)
                    else:
                        raise TypeError("Unsupported type for merging")
        return merged_result
    elif isinstance(results[0], list):
        # Merge lists
        merged_result = []
        for result in results:
            merged_result.extend(result)
        return merged_result
    else:
        raise TypeError("Unsupported return type for merging")


def concurrent_execute(n_thread=5):
    def decorator(func):
        def wrapper(data_list, *args, **kwargs):
            print(f"Splitting {len(data_list)} items into {n_thread} threads")
            split_data = split_list(data_list, n_thread)
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(func, split_data[i], *args, **kwargs) for i in range(len(split_data))]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            return merge_results(results)
        return wrapper
    return decorator