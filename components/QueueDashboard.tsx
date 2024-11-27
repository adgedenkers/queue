import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { Loader2 } from 'lucide-react';

const APIDashboard = () => {
  // State for authentication
  const [userId] = useState('1');
  const [token] = useState('6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b');

  // State for queue entry
  const [queueText, setQueueText] = useState('');
  
  // State for shoe entry
  const [shoeData, setShoeData] = useState({
    brand: '',
    model: '',
    size: '',
    color: '',
    gender: 'Mens'
  });

  // State for API interaction
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleQueueSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      console.log('Submitting queue item...');
      const response = await fetch('https://api.denkers.co/queue/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-token': token,
          'user-id': userId
        },
        body: JSON.stringify({
          user_id: parseInt(userId),
          raw_text: queueText,
          options: {},
          properties: {},
          images: []
        })
      });
      
      const data = await response.json();
      console.log('Queue response:', data);
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to create queue item');
      }
      
      setResult(data);
      setQueueText(''); // Clear the input on success
    } catch (err) {
      console.error('Queue error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleShoeSubmit = async () => {
    if (!shoeData.brand || !shoeData.model || !shoeData.size) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      console.log('Submitting shoe data...', shoeData);
      const response = await fetch('https://api.denkers.co/shoes/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-token': token,
          'user-id': userId
        },
        body: JSON.stringify({
          ...shoeData,
          size: parseFloat(shoeData.size)
        })
      });
      
      const data = await response.json();
      console.log('Shoe response:', data);
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to create shoe record');
      }
      
      setResult(data);
      // Clear form on success
      setShoeData({
        brand: '',
        model: '',
        size: '',
        color: '',
        gender: 'Mens'
      });
    } catch (err) {
      console.error('Shoe error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4 space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Queue Entry</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              value={queueText}
              onChange={(e) => setQueueText(e.target.value)}
              placeholder="Enter queue text..."
              className="h-32"
            />
            <Button 
              onClick={handleQueueSubmit}
              disabled={loading || !queueText}
              className="w-full"
            >
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : 'Submit to Queue'}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Shoe Entry</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              value={shoeData.brand}
              onChange={(e) => setShoeData({...shoeData, brand: e.target.value})}
              placeholder="Brand"
            />
            <Input
              value={shoeData.model}
              onChange={(e) => setShoeData({...shoeData, model: e.target.value})}
              placeholder="Model"
            />
            <div className="grid grid-cols-2 gap-4">
              <Input
                value={shoeData.size}
                onChange={(e) => setShoeData({...shoeData, size: e.target.value})}
                placeholder="Size"
                type="number"
                step="0.5"
              />
              <select
                value={shoeData.gender}
                onChange={(e) => setShoeData({...shoeData, gender: e.target.value})}
                className="p-2 border rounded"
              >
                <option>Mens</option>
                <option>Womens</option>
                <option>Unisex</option>
                <option>Youth</option>
                <option>Kids</option>
              </select>
            </div>
            <Input
              value={shoeData.color}
              onChange={(e) => setShoeData({...shoeData, color: e.target.value})}
              placeholder="Color"
            />
            <Button 
              onClick={handleShoeSubmit}
              disabled={loading || !shoeData.brand || !shoeData.model || !shoeData.size}
              className="w-full"
            >
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : 'Submit Shoe'}
            </Button>
          </CardContent>
        </Card>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="bg-gray-100 p-4 rounded overflow-auto">
              {JSON.stringify(result, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default APIDashboard;