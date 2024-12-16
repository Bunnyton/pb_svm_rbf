package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"
)

// Data структура для хранения данных
type Data struct {
	XTrain [][]float64
	yTrain []float64
	XTest  [][]float64
	yTest  []float64
}

// Загрузка и разделение данных
func loadAndSplitData(filePath string, trainRatio float64) (Data, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return Data{}, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return Data{}, err
	}

	// Парсим данные из CSV
	var features [][]float64
	var labels []float64
	for i, row := range rows {
		if i == 0 {
			continue // пропускаем заголовок
		}
		feature := make([]float64, len(row)-1)
		for j := range row[:len(row)-1] {
			feature[j], err = strconv.ParseFloat(row[j], 64)
			if err != nil {
				log.Fatalf("Ошибка при парсинге данных: %v", err)
			}
		}
		label, err := strconv.ParseFloat(row[len(row)-1], 64)
		if err != nil {
			log.Fatalf("Ошибка при парсинге меток: %v", err)
		}

		features = append(features, feature)
		labels = append(labels, label)
	}

	// Разделяем данные на обучающую и тестовую выборки
	rand.Seed(time.Now().UnixNano())
	indices := rand.Perm(len(features))
	trainSize := int(float64(len(features)) * trainRatio)

	XTrain := make([][]float64, trainSize)
	yTrain := make([]float64, trainSize)
	XTest := make([][]float64, len(features)-trainSize)
	yTest := make([]float64, len(features)-trainSize)

	for i := 0; i < len(indices); i++ {
		if i < trainSize {
			XTrain[i] = features[indices[i]]
			yTrain[i] = labels[indices[i]]
		} else {
			XTest[i-trainSize] = features[indices[i]]
			yTest[i-trainSize] = labels[indices[i]]
		}
	}

	return Data{
		XTrain: XTrain,
		yTrain: yTrain,
		XTest:  XTest,
		yTest:  yTest,
	}, nil
}

// Простая линейная модель SVM
func simpleSVMTrain(X [][]float64, y []float64) ([]float64, float64) {
	// Обучение линейного SVM, используя перцептронный алгоритм
	nFeatures := len(X[0])
	weights := make([]float64, nFeatures)
	bias := 0.0
	learningRate := 0.1
	epochs := 50

	// Простая оптимизация
	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < len(X); i++ {
			pred := 0.0
			for j := 0; j < nFeatures; j++ {
				pred += weights[j] * X[i][j]
			}
			pred += bias
			if y[i]*pred <= 0 { // Ошибка
				for j := 0; j < nFeatures; j++ {
					weights[j] += learningRate * y[i] * X[i][j]
				}
				bias += learningRate * y[i]
			}
		}
	}

	return weights, bias
}

func svmPredict(X [][]float64, weights []float64, bias float64) []float64 {
	// Предсказания на основе весов и модели
	predictions := make([]float64, len(X))
	for i := range X {
		sum := bias
		for j := range weights {
			sum += weights[j] * X[i][j]
		}
		predictions[i] = sum
	}
	return predictions
}

func main() {
	data, err := loadAndSplitData("WineQT.csv", 0.5)
	if err != nil {
		log.Fatalf("Ошибка загрузки данных: %v", err)
	}

	fmt.Println("Данные загружены. Начинаем обучение модели...")

	weights, bias := simpleSVMTrain(data.XTrain, data.yTrain)
	fmt.Printf("Обучение завершено, bias = %.2f\n", bias)

	predictions := svmPredict(data.XTest, weights, bias)

	correct := 0
	for i, p := range predictions {
		fmt.Printf("%s ", i)
		fmt.Printf("%s ", p)
		fmt.Printf("%s\n", data.yTest[i])
		if (p == data.yTest[i]) {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(data.yTest)) * 100
	fmt.Printf("Точность на тестовой выборке: %.2f%%\n", accuracy)
}

