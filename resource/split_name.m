load("E:\\ubuntu\\dataset\\dish\\Food-475-TEST.mat")
id_size_matirx = size(id);
id_size = id_size_matirx(1);

total_name = [];
num = 0;
for i = 1:id_size
    id_name_array = id(i);
    id_name = id_name_array{1};
    id_paths = images(id_name);
    paths_size_matirx = size(id_paths);
    for k = 1:paths_size_matirx(2)
        path_array = id_paths(k);
        path = path_array{1};
        if ~contains(path,'food50')
            path;
            num = num + 1
            total_name = horzcat(total_name, (id_name + "," + path));
        end
    end
end
