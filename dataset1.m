num_samples = 1000;
C1 = zeros(num_samples, 2);
C2 = zeros(num_samples, 2);
inside_semicircle = @(x, y, center, radius, left_half) ...
    (x - center(1)).^2 + (y - center(2)).^2 <= radius^2 & ...
    (left_half & x <= center(1) | ~left_half & x >= center(1));

%Area of C1
i = 1;
while i <= num_samples
    x = -2 + 4*rand;
    y = -3 + 5*rand;
    if inside_semicircle(x, y, [0, 0], 2, true) && ...
       ~inside_semicircle(x, y, [0, 0], 1, true)
        C1(i, :) = [x y];
        i = i + 1;
    elseif inside_semicircle(x, y, [0, -1], 1, false)
        C1(i, :) = [x y];
        i = i + 1;
    end
end

% Area of C2
j = 1;
while j <= num_samples
    x = -2 + 4*rand;
    y = -3 + 5*rand;
    if inside_semicircle(x, y, [0, -1], 2, false) && ...
       ~inside_semicircle(x, y, [0, -1], 1, false)
        C2(j, :) = [x y];
        j = j + 1;
    elseif inside_semicircle(x, y, [0, 0], 1, true)
        C2(j, :) = [x y];
        j = j + 1;
    end
end
figure;
scatter(C1(:,1), C1(:,2), 'r.');
hold on;
scatter(C2(:,1), C2(:,2), 'b.');
axis equal;
xlim([-3, 3]);
ylim([-3, 3]);
title('Dataset C1 and C2');
hold off;
combined_data = [C1; C2];
labels = [ones(num_samples, 1); zeros(num_samples, 1)];
combined_data_with_labels = [combined_data labels];
csvwrite('dataset.csv', combined_data_with_labels);
rand_indices = randperm(size(combined_data_with_labels, 1));
shuffled_data_with_labels = combined_data_with_labels(rand_indices, :);
num_train = floor(size(shuffled_data_with_labels, 1) * 0.8);
num_test = size(shuffled_data_with_labels, 1) - num_train;
train_data = shuffled_data_with_labels(1:num_train, :);
test_data = shuffled_data_with_labels(num_train+1:end, :);
csvwrite('train_dataset.csv', train_data);
csvwrite('test_dataset.csv', test_data);