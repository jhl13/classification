load("E:\\ubuntu\\classification\\test.mat");
name_size_matrix = size(total_name_t);
name_size = name_size_matrix(1)

f = fopen("E:\\ubuntu\\classification\\test.txt", 'w+');
for i = 1:name_size
    i
    fprintf(f, total_name_t(i));
    fprintf(f, '\n');
end
fclose(f);
