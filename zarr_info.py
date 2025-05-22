import zarr

def print_arrays(store, path=''):
    for key in store:
        obj = store[key]
        obj_path = f"{path}/{key}" if path else key
        if isinstance(obj, zarr.hierarchy.Group):
            # 递归遍历子组
            print_arrays(obj, obj_path)
        else:
            # 这是数组，打印路径和大小
            print(f"数组名称: {obj_path}，大小: {obj.shape}")

if __name__ == '__main__':
    zarr_file_path = '/data1/lxy-24/Data/Maniskill/unimix/test/metaworld_basketball_expert.zarr'  # 替换成你的 Zarr 文件路径
    zarr_store = zarr.open(zarr_file_path, mode='r')
    print_arrays(zarr_store)
