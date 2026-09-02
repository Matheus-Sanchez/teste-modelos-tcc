-- Audit companion generated from investigacao_ativacoes_atualizada.json.
-- Raw manifests/CSV/NumPy files remain the primary evidence.
CREATE TABLE activation_runs (
  root_id TEXT, dataset TEXT, activation TEXT, extra_fraction REAL,
  test_accuracy REAL, test_macro_f1 REAL, test_loss REAL,
  predicted_class_count INTEGER, predicted_mode_share REAL,
  logit_component_std REAL
);
INSERT INTO activation_runs VALUES
  ('augmentation_2_0', 'fashion_mnist', 'relu', 2.0, 0.876, 0.870729984236, 0.31127589941, 10, 0.127047619048, 4.980807801716),
  ('augmentation_2_0', 'fashion_mnist', 'sigmoid', 2.0, 0.918095238095, 0.91791175673, 0.223579868674, 10, 0.10580952381, 6.370195172721),
  ('augmentation_2_0', 'mnist', 'relu', 2.0, 0.987426176415, 0.987190471562, 0.052007563412, 10, 0.112497618594, 6.56775238438),
  ('augmentation_2_0', 'mnist', 'sigmoid', 2.0, 0.989236044961, 0.989162898279, 0.03717764467, 10, 0.112592874833, 5.220622597309),
  ('augmentation_2_0', 'mnist', 'softmax', 2.0, 0.112497618594, 0.020224334275, 2.301202058792, 1, 1.0, 3.9369e-08),
  ('augmentation_0_5', 'fashion_mnist', 'relu', 0.5, 0.87819047619, 0.875009941372, 0.327285259962, 10, 0.132095238095, 3.487454152381),
  ('augmentation_0_5', 'fashion_mnist', 'sigmoid', 0.5, 0.900476190476, 0.900415028206, 0.277063459158, 10, 0.115238095238, 5.476228433189),
  ('augmentation_0_5', 'fashion_mnist', 'softmax', 0.5, 0.1, 0.018181818182, 2.302587509155, 1, 1.0, 1.2e-11),
  ('augmentation_0_5', 'kmnist', 'relu', 0.5, 0.97019047619, 0.970159630162, 0.138363972306, 10, 0.10419047619, 16.264376311866),
  ('augmentation_0_5', 'kmnist', 'sigmoid', 0.5, 0.971142857143, 0.971143229199, 0.106368862092, 10, 0.102476190476, 3.758276488012),
  ('augmentation_0_5', 'mnist', 'relu', 0.5, 0.983711183082, 0.983603280405, 0.059442941099, 10, 0.11440274338, 4.467336233782),
  ('augmentation_0_5', 'mnist', 'sigmoid', 0.5, 0.984568489236, 0.984366446096, 0.055624689907, 10, 0.112783387312, 4.515069414822),
  ('augmentation_0_5', 'mnist', 'softmax', 0.5, 0.112497618594, 0.020224334275, 2.301443815231, 1, 1.0, 9e-12);
