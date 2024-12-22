<template>
  <div class="drug-category">
    <h2 class="title">Classification</h2>
    <el-card shadow="always">
      <el-form :model="form" label-width="120px">
        <!-- Input for structureId -->
        <el-form-item label="structureId">
          <el-input
            type="textarea"
            v-model="form.structureId"
            placeholder="Enter structureId to classify"
            class="input-text"
          ></el-input>
        </el-form-item>
        <!-- Input for chainId -->
        <el-form-item label="chainId">
          <el-input
            type="textarea"
            v-model="form.chainId"
            placeholder="Enter chainId to classify"
            class="input-text"
          ></el-input>
        </el-form-item>
        <!-- Input for sequence -->
        <el-form-item label="sequence">
          <el-input
            type="textarea"
            v-model="form.sequence"
            placeholder="Enter sequence to classify"
            class="input-text"
          ></el-input>
        </el-form-item>
        <!-- Input for residueCount -->
        <el-form-item label="residueCount">
          <el-input
            type="textarea"
            v-model="form.residueCount"
            placeholder="Enter residueCount to classify"
            class="input-text"
          ></el-input>
        </el-form-item>
        <!-- Dropdown for selecting a model -->
        <el-form-item label="Select Model">
          <el-select v-model="form.model" placeholder="Select a model" class="model-select">
            <el-option
              v-for="model in models"
              :key="model"
              :label="model"
              :value="model"
            ></el-option>
          </el-select>
        </el-form-item>
        <!-- Classify button -->
        <el-form-item class="button-container">
          <el-button
            type="primary"
            size="large"
            @click="classifyText"
            :loading="loading"
          >Classify</el-button>
        </el-form-item>
        <!-- Display classification result -->
        <el-form-item v-if="result" label="Classification Result">
          <el-input
            type="textarea"
            :value="result"
            readonly
            class="result-box"
          ></el-input>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "DrugCategory",
  data() {
    return {
      form: {
        structureId: "", 
        chainId: "",
        sequence: "",
        residueCount: "",
        model: "",
      },
      models: ["cnn", "lstm", "rnn", "transformer", "kmeans", "random_forest"], // Available model options
      result: "", // Classification result
      loading: false, // Loading state for the classify button
    };
  },
  methods: {
    async classifyText() {
      // Ensure all fields are filled before submitting
      if (!this.form.structureId || !this.form.chainId || !this.form.sequence || !this.form.residueCount || !this.form.model) {
        this.$message.error("Please fill in both the text and select a model.");
        return;
      }
      this.loading = true;
      try {
        // Send classification request to the backend
        const response = await axios.post("http://localhost:5001/api/classify", {
          structureId: this.form.structureId,
          chainId: this.form.chainId,
          sequence: this.form.sequence,
          residueCount: this.form.residueCount,
          model: this.form.model,
        });
        this.result = response.data.result;
      } catch (error) {
        console.error("Error while classifying text:", error);
        this.$message.error("Classification failed. Please try again.");
      } finally {
        this.loading = false;
      }
    },
  },
};
</script>

<style scoped>
.drug-category {
  max-width: 600px;
  margin: 50px auto;
  padding: 20px;
}

.title {
  text-align: center;
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 20px;
  color: #2c3e50;
}

.el-card {
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

.input-text {
  min-height: 100px;
}

.model-select {
  width: 100%;
}

.button-container {
  text-align: center;
  margin-top: 20px;
}

.result-box {
  background-color: #f9f9f9;
  min-height: 80px;
  color: #2c3e50;
  border: 1px solid #dcdfe6;
  border-radius: 5px;
  padding: 10px;
}
</style>
