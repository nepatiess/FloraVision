import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const FloraVisionApp());
}

class FloraVisionApp extends StatelessWidget {
  const FloraVisionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'FloraVision',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF7A9E7E),
        ),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();

  XFile? _selectedImage;

  String? _prediction;
  double? _confidence;

  bool _isLoading = false;

  // Galeriden fotoğraf seçer.
  Future<void> _pickFromGallery() async {
    final XFile? image = await _picker.pickImage(
      source: ImageSource.gallery,
    );

    if (image == null) {
      return;
    }

    setState(() {
      _selectedImage = image;
      _prediction = null;
      _confidence = null;
    });
  }

  // Kamerayı açar ve fotoğraf çeker.
  Future<void> _takePhoto() async {
    final XFile? image = await _picker.pickImage(
      source: ImageSource.camera,
    );

    if (image == null) {
      return;
    }

    setState(() {
      _selectedImage = image;
      _prediction = null;
      _confidence = null;
    });
  }

  // Seçilen fotoğrafı FastAPI'ye gönderir.
  Future<void> _predictFlower() async {
    if (_selectedImage == null) {
      return;
    }

    setState(() {
      _isLoading = true;
      _prediction = null;
      _confidence = null;
    });

    try {
      final uri = Uri.parse(
        'http://10.0.2.2:8000/predict',
      );

      final request = http.MultipartRequest(
        'POST',
        uri,
      );

      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          _selectedImage!.path,
        ),
      );

      final response = await request.send();

      final responseBody =
          await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final data = jsonDecode(responseBody);

        setState(() {
          _prediction = data['prediction'];
          _confidence =
              (data['confidence'] as num).toDouble();
        });
      } else {
        throw Exception(
          'Prediction failed: ${response.statusCode}',
        );
      }
    } catch (error) {
      debugPrint('Prediction error: $error');

      if (!mounted) {
        return;
      }

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Could not connect to FloraVision API.',
          ),
        ),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF7F8F3),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Spacer(),

              // Fotoğraf seçilmediyse çiçek ikonu göster.
              // Seçildiyse seçilen fotoğrafı göster.
              if (_selectedImage == null)
                const Icon(
                  Icons.local_florist_rounded,
                  size: 90,
                  color: Color(0xFF7A9E7E),
                )
              else
                ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Image.file(
                    File(_selectedImage!.path),
                    height: 250,
                    fit: BoxFit.cover,
                  ),
                ),

              const SizedBox(height: 24),

              const Text(
                'FloraVision',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 36,
                  fontWeight: FontWeight.bold,
                ),
              ),

              const SizedBox(height: 10),

              const Text(
                'Discover flowers with artificial intelligence.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.black54,
                ),
              ),

              // Tahmin sonucu
              if (_prediction != null) ...[
                const SizedBox(height: 20),

                Text(
                  _prediction!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 26,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF347548),
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  'Confidence: ${_confidence!.toStringAsFixed(2)}%',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 16,
                    color: Colors.black54,
                  ),
                ),
              ],

              const Spacer(),

              // Kamera
              FilledButton.icon(
                onPressed: _isLoading ? null : _takePhoto,
                icon: const Icon(
                  Icons.camera_alt_rounded,
                ),
                label: const Text('Take a Photo'),
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    vertical: 16,
                  ),
                ),
              ),

              const SizedBox(height: 12),

              // Galeri
              OutlinedButton.icon(
                onPressed:
                    _isLoading ? null : _pickFromGallery,
                icon: const Icon(
                  Icons.photo_library_rounded,
                ),
                label: const Text(
                  'Choose from Gallery',
                ),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    vertical: 16,
                  ),
                ),
              ),

              // Fotoğraf seçilince Analyze butonu göster.
              if (_selectedImage != null) ...[
                const SizedBox(height: 12),

                FilledButton.icon(
                  onPressed:
                      _isLoading ? null : _predictFlower,
                  icon: _isLoading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                          ),
                        )
                      : const Icon(
                          Icons.auto_awesome_rounded,
                        ),
                  label: Text(
                    _isLoading
                        ? 'Analyzing...'
                        : 'Identify Flower',
                  ),
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                      vertical: 16,
                    ),
                  ),
                ),
              ],

              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}